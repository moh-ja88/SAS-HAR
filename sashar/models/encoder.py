"""
SAS-HAR Encoder Module
CNN Feature Encoder + Transformer Temporal Module
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class DepthwiseSeparableConv1d(nn.Module):
    """
    Depthwise Separable Convolution for efficiency
    
    Reduces parameters by 8-10x compared to standard convolutions
    while maintaining similar representational power.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False
    ):
        super().__init__()
        
        # Depthwise: spatial convolution per channel
        self.depthwise = nn.Conv1d(
            in_channels, in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,  # Key: separate per channel
            bias=bias
        )
        
        # Pointwise: 1x1 convolution for channel mixing
        self.pointwise = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=1,
            bias=bias
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class CNNFeatureEncoder(nn.Module):
    """
    CNN Feature Encoder for local spatial feature extraction
    
    Uses depthwise separable convolutions for efficiency.
    
    Architecture:
        Input: [Batch, Channels, Time]
        Block 1: Conv (6 -> 64) + BN + ReLU + MaxPool
        Block 2: Conv (64 -> 128) + BN + ReLU + MaxPool  
        Block 3: Conv (128 -> 256) + BN + ReLU + MaxPool
        Output: [Batch, 256, Time/8]
    """
    
    def __init__(
        self,
        input_channels: int = 6,
        hidden_dims: list = [64, 128, 256],
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Block 1: 6 -> 64
        self.block1 = nn.Sequential(
            DepthwiseSeparableConv1d(input_channels, hidden_dims[0], kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.MaxPool1d(kernel_size=2, stride=2)  # T/2
        )
        
        # Block 2: 64 -> 128
        self.block2 = nn.Sequential(
            DepthwiseSeparableConv1d(hidden_dims[0], hidden_dims[1], kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.MaxPool1d(kernel_size=2, stride=2)  # T/4
        )
        
        # Block 3: 128 -> 256
        self.block3 = nn.Sequential(
            DepthwiseSeparableConv1d(hidden_dims[1], hidden_dims[2], kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dims[2]),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.MaxPool1d(kernel_size=2, stride=2)  # T/8
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [Batch, Channels, Time]
        
        Returns:
            Features [Batch, 256, Time/8]
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x


class EfficientLinearAttention(nn.Module):
    """
    Efficient Linear Attention with O(n) complexity
    
    Standard attention: O(n²) - prohibitive for long sequences
    Linear attention: O(n) - enables processing of long temporal sequences
    
    Based on: "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention"
    (Katharopoulos et al., 2020)
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Projections
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Input tensor [Batch, Time, Dim]
            return_attention: Whether to return attention weights
        
        Returns:
            output: [Batch, Time, Dim]
            attention_weights: [Batch, Heads, Time, Time] (if return_attention)
        """
        B, T, C = x.shape
        
        # Project to Q, K, V
        Q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        # Q, K, V: [Batch, Heads, Time, HeadDim]
        
        # Apply kernel function (elu + 1) for positive values
        Q = F.elu(Q) + 1
        K = F.elu(K) + 1
        
        # Linear attention computation
        # Standard: softmax(Q @ K.T) @ V  -> O(n²)
        # Linear: Q @ (K.T @ V) / (Q @ K.T @ 1)  -> O(n)
        
        # Compute K^T @ V: [Batch, Heads, HeadDim, HeadDim]
        KV = torch.einsum('bhnd,bhne->bhde', K, V)
        
        # Compute Q @ (K^T @ V): [Batch, Heads, Time, HeadDim]
        Q_KV = torch.einsum('bhnd,bhde->bhne', Q, KV)
        
        # Compute normalizer: Q @ K^T @ 1
        K_sum = K.sum(dim=2, keepdim=True)  # [Batch, Heads, 1, HeadDim]
        normalizer = torch.einsum('bhnd,bhkd->bhnk', Q, K_sum).squeeze(-1)  # [Batch, Heads, Time]
        
        # Normalize
        out = Q_KV / (normalizer.unsqueeze(-1) + 1e-6)
        
        # Reshape back
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection
        out = self.out_proj(out)
        out = self.dropout(out)
        
        if return_attention:
            # Approximate attention for visualization (not exact)
            attention = torch.einsum('bhnd,bhmd->bhnm', Q, K) / (normalizer.unsqueeze(-1) + 1e-6)
            return out, attention
        else:
            return out, None


class TransformerBlock(nn.Module):
    """
    Transformer Block with Pre-Norm architecture
    
    Components:
    - Multi-head self-attention (efficient linear)
    - Feed-forward network
    - Layer normalization
    - Residual connections
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        mlp_ratio: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Pre-norm
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientLinearAttention(dim, num_heads, dropout)
        
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout)
        )
        
    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Input tensor [Batch, Time, Dim]
            return_attention: Whether to return attention weights
        
        Returns:
            output: [Batch, Time, Dim]
            attention: [Batch, Heads, Time, Time] (if return_attention)
        """
        # Self-attention with residual
        attn_out, attention = self.attn(self.norm1(x), return_attention)
        x = x + attn_out
        
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        
        return x, attention


class TransformerTemporalModule(nn.Module):
    """
    Transformer Temporal Module for long-range dependency modeling
    
    Stacks multiple transformer blocks with optional positional encoding.
    """
    
    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 3,
        mlp_ratio: int = 2,
        dropout: float = 0.1,
        max_seq_len: int = 1024
    ):
        super().__init__()
        
        # Learnable positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(1, max_seq_len, dim) * 0.02
        )
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(dim)
        
    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, list]:
        """
        Args:
            x: Input tensor [Batch, Time, Dim]
            return_attention: Whether to return attention weights
        
        Returns:
            output: [Batch, Time, Dim]
            attentions: List of attention tensors (if return_attention)
        """
        B, T, C = x.shape
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :T, :]
        
        # Apply transformer layers
        attentions = []
        for layer in self.layers:
            x, attn = layer(x, return_attention)
            if return_attention:
                attentions.append(attn)
        
        # Final normalization
        x = self.norm(x)
        
        return x, attentions


# Testing
if __name__ == "__main__":
    # Test CNN Encoder
    encoder = CNNFeatureEncoder(input_channels=6)
    x = torch.randn(4, 6, 128)
    
    features = encoder(x)
    print(f"CNN Encoder:")
    print(f"  Input: {x.shape}")
    print(f"  Output: {features.shape}")
    print(f"  Parameters: {sum(p.numel() for p in encoder.parameters()):,}")
    
    # Test Transformer
    transformer = TransformerTemporalModule(dim=256, num_heads=4, num_layers=3)
    x = torch.randn(4, 16, 256)  # [B, T/8, 256]
    
    output, attentions = transformer(x, return_attention=True)
    print(f"\nTransformer:")
    print(f"  Input: {x.shape}")
    print(f"  Output: {output.shape}")
    print(f"  Parameters: {sum(p.numel() for p in transformer.parameters()):,}")
    
    # Test full encoder pipeline
    x = torch.randn(4, 6, 128)
    cnn_features = encoder(x)  # [4, 256, 16]
    cnn_features = cnn_features.transpose(1, 2)  # [4, 16, 256]
    temporal_features, _ = transformer(cnn_features)
    
    print(f"\nFull Encoder Pipeline:")
    print(f"  Input: {x.shape}")
    print(f"  CNN Output: {cnn_features.shape}")
    print(f"  Transformer Output: {temporal_features.shape}")
    print(f"  Total Parameters: {sum(p.numel() for p in encoder.parameters()) + sum(p.numel() for p in transformer.parameters()):,}")
