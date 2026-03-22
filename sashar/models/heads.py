"""
Task-specific heads for SAS-HAR model.

This module implements:
- BoundaryHead: Detects activity boundaries in temporal sequences
- ClassificationHead: Classifies activity type for segments
- MultiTaskHead: Combined boundary + classification prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


class BoundaryHead(nn.Module):
    """
    Boundary detection head with temporal modeling.
    
    Predicts boundary probability at each time step.
    Uses temporal convolution for smoothing and context.
    """
    
    def __init__(
        self,
        d_model: int = 512,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
        use_temporal_conv: bool = True,
        kernel_size: int = 5
    ):
        super().__init__()
        
        self.use_temporal_conv = use_temporal_conv
        
        # MLP layers
        layers = []
        in_dim = d_model
        for i in range(num_layers - 1):
            out_dim = hidden_dim // (2 ** i) if i > 0 else hidden_dim
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            in_dim = out_dim
        
        # Final prediction layer
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)
        
        # Temporal smoothing convolution
        if use_temporal_conv:
            self.temporal_conv = nn.Sequential(
                nn.Conv1d(1, 16, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.GELU(),
                nn.Conv1d(16, 1, kernel_size=kernel_size, padding=kernel_size // 2)
            )
        
        # Boundary-aware attention for refinement
        self.boundary_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # Learnable boundary query
        self.boundary_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
    
    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Input features of shape (batch, seq_len, d_model)
            return_attention: Whether to return attention weights
        
        Returns:
            boundary_scores: Boundary probabilities of shape (batch, seq_len)
            attention_weights: Optional attention weights of shape (batch, seq_len)
        """
        B, T, D = x.shape
        
        # Boundary query attention
        query = self.boundary_query.expand(B, -1, -1)  # (B, 1, D)
        attn_out, attn_weights = self.boundary_attention(
            query=query,
            key=x,
            value=x
        )  # (B, 1, D), (B, 1, T)
        
        # Combine with input
        x_refined = x + 0.1 * attn_out.expand(-1, T, -1)
        
        # MLP prediction
        scores = self.mlp(x_refined).squeeze(-1)  # (B, T)
        
        # Temporal smoothing
        if self.use_temporal_conv:
            scores = self.temporal_conv(scores.unsqueeze(1)).squeeze(1)  # (B, T)
        
        # Sigmoid for probability
        boundary_scores = torch.sigmoid(scores)
        
        if return_attention:
            return boundary_scores, attn_weights.squeeze(1)
        
        return boundary_scores, None


class ClassificationHead(nn.Module):
    """
    Activity classification head.
    
    Uses global pooling followed by MLP classifier.
    Supports both segment-level and window-level classification.
    """
    
    def __init__(
        self,
        d_model: int = 512,
        num_classes: int = 6,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
        pooling: str = 'avg'  # 'avg', 'max', 'attention'
    ):
        super().__init__()
        
        self.pooling = pooling
        self.num_classes = num_classes
        
        # Pooling layer
        if pooling == 'attention':
            self.attention_pool = nn.Sequential(
                nn.Linear(d_model, d_model // 4),
                nn.Tanh(),
                nn.Linear(d_model // 4, 1)
            )
        
        # Classifier MLP
        layers = []
        in_dim = d_model
        
        for i in range(num_layers):
            out_dim = hidden_dim // (2 ** i) if i < num_layers - 1 else num_classes
            layers.append(nn.Linear(in_dim, out_dim))
            
            if i < num_layers - 1:
                layers.extend([
                    nn.LayerNorm(out_dim),
                    nn.GELU(),
                    nn.Dropout(dropout)
                ])
            
            in_dim = out_dim
        
        self.classifier = nn.Sequential(*layers)
    
    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply pooling over temporal dimension.
        
        Args:
            x: (batch, seq_len, d_model)
        
        Returns:
            pooled: (batch, d_model)
        """
        if self.pooling == 'avg':
            return x.mean(dim=1)
        elif self.pooling == 'max':
            return x.max(dim=1)[0]
        elif self.pooling == 'attention':
            # Compute attention weights
            attn_weights = self.attention_pool(x)  # (B, T, 1)
            attn_weights = F.softmax(attn_weights, dim=1)
            
            # Weighted sum
            return (x * attn_weights).sum(dim=1)  # (B, D)
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features of shape (batch, seq_len, d_model)
        
        Returns:
            logits: Classification logits of shape (batch, num_classes)
        """
        # Pool over time
        pooled = self._pool(x)  # (B, D)
        
        # Classify
        logits = self.classifier(pooled)  # (B, num_classes)
        
        return logits


class MultiTaskHead(nn.Module):
    """
    Combined boundary detection and classification head.
    
    Enables joint training for segmentation and recognition.
    """
    
    def __init__(
        self,
        d_model: int = 512,
        num_classes: int = 6,
        boundary_hidden: int = 256,
        class_hidden: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.boundary_head = BoundaryHead(
            d_model=d_model,
            hidden_dim=boundary_hidden,
            dropout=dropout
        )
        
        self.classification_head = ClassificationHead(
            d_model=d_model,
            num_classes=num_classes,
            hidden_dim=class_hidden,
            dropout=dropout
        )
    
    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Input features of shape (batch, seq_len, d_model)
            return_attention: Whether to return boundary attention
        
        Returns:
            Dictionary containing:
                - boundary_scores: (batch, seq_len)
                - class_logits: (batch, num_classes)
                - boundary_attention: (batch, seq_len) if return_attention
        """
        boundary_scores, boundary_attention = self.boundary_head(
            x, return_attention=return_attention
        )
        class_logits = self.classification_head(x)
        
        output = {
            'boundary_scores': boundary_scores,
            'class_logits': class_logits
        }
        
        if return_attention:
            output['boundary_attention'] = boundary_attention
        
        return output


class SegmentClassificationHead(nn.Module):
    """
    Segment-level classification using boundary information.
    
    Uses detected boundaries to create segments and classify each.
    """
    
    def __init__(
        self,
        d_model: int = 512,
        num_classes: int = 6,
        boundary_threshold: float = 0.5
    ):
        super().__init__()
        
        self.boundary_threshold = boundary_threshold
        
        self.classifier = ClassificationHead(
            d_model=d_model,
            num_classes=num_classes,
            pooling='attention'
        )
    
    def forward(
        self,
        x: torch.Tensor,
        boundary_scores: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input features of shape (batch, seq_len, d_model)
            boundary_scores: Optional boundary scores of shape (batch, seq_len)
        
        Returns:
            logits: Classification logits of shape (batch, num_classes)
        """
        if boundary_scores is not None:
            # Use boundary scores to weight features
            # Higher weight for non-boundary regions
            weights = 1.0 - boundary_scores  # (B, T)
            weights = weights.unsqueeze(-1)  # (B, T, 1)
            weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
            
            # Weighted features
            x = x * weights
        
        return self.classifier(x)


class TransitionalActivityHead(nn.Module):
    """
    Specialized head for transitional activity detection.
    
    Detects and classifies transitional activities (e.g., walk→run).
    """
    
    def __init__(
        self,
        d_model: int = 512,
        num_transitions: int = 10,
        hidden_dim: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Transition detector (binary: is this a transition point?)
        self.transition_detector = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Transition classifier (which transition type?)
        self.transition_classifier = nn.Sequential(
            nn.Linear(d_model * 2, hidden_dim),  # Concatenate before/after features
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_transitions)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        boundary_indices: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Input features of shape (batch, seq_len, d_model)
            boundary_indices: Optional boundary positions
        
        Returns:
            Dictionary containing:
                - transition_scores: (batch, seq_len) probability of transition
                - transition_logits: (batch, seq_len, num_transitions)
        """
        # Detect transitions
        transition_scores = self.transition_detector(x).squeeze(-1)  # (B, T)
        
        # Classify transition types at each position
        # Shift x to get before/after context
        x_before = F.pad(x[:, :-1, :], (0, 0, 1, 0), value=0)  # (B, T, D)
        x_after = F.pad(x[:, 1:, :], (0, 0, 0, 1), value=0)  # (B, T, D)
        
        x_combined = torch.cat([x_before, x_after], dim=-1)  # (B, T, 2D)
        transition_logits = self.transition_classifier(x_combined)  # (B, T, num_transitions)
        
        return {
            'transition_scores': transition_scores,
            'transition_logits': transition_logits
        }


# Loss functions for heads

class BoundaryLoss(nn.Module):
    """
    Loss function for boundary detection.
    
    Combines BCE with focal loss for class imbalance.
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        pos_weight: Optional[float] = None
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred: Predicted boundary scores (B, T)
            target: Ground truth boundaries (B, T)
        
        Returns:
            loss: Scalar loss value
        """
        # Binary cross-entropy
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        
        # Focal weight
        pt = torch.where(target == 1, pred, 1 - pred)
        focal_weight = (1 - pt) ** self.gamma
        
        # Class balance weight
        if self.pos_weight is not None:
            weight = torch.where(target == 1, self.pos_weight, 1.0)
            bce = bce * weight
        
        # Combine
        loss = self.alpha * focal_weight * bce
        
        return loss.mean()


class ConsistencyLoss(nn.Module):
    """
    Consistency loss between boundaries and activity changes.
    
    Encourages boundaries at activity transitions.
    """
    
    def __init__(self, margin: float = 0.1):
        super().__init__()
        self.margin = margin
    
    def forward(
        self,
        boundary_scores: torch.Tensor,
        class_logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            boundary_scores: Predicted boundaries (B, T)
            class_logits: Class predictions (B, T, C) or (B, C)
            labels: Ground truth labels (B, T) or (B,)
        
        Returns:
            loss: Scalar consistency loss
        """
        # Get predicted class at each time step
        if class_logits.dim() == 2:
            # Global classification - expand to time steps
            pred_classes = class_logits.argmax(dim=-1).unsqueeze(1).expand(-1, labels.size(1))
        else:
            pred_classes = class_logits.argmax(dim=-1)
        
        # Find activity changes
        label_changes = (labels[:, 1:] != labels[:, :-1]).float()  # (B, T-1)
        
        # Boundary scores at change points
        boundary_at_changes = boundary_scores[:, 1:]  # (B, T-1)
        
        # Loss: encourage high boundary scores at changes
        # Hinge loss
        loss = F.relu(self.margin - boundary_at_changes * label_changes)
        
        return loss.mean()


if __name__ == "__main__":
    # Test the heads
    batch_size = 4
    seq_len = 128
    d_model = 512
    num_classes = 6
    
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Test BoundaryHead
    print("Testing BoundaryHead...")
    boundary_head = BoundaryHead(d_model)
    boundary_scores, attn = boundary_head(x, return_attention=True)
    print(f"  Boundary scores shape: {boundary_scores.shape}")
    print(f"  Attention shape: {attn.shape}")
    
    # Test ClassificationHead
    print("\nTesting ClassificationHead...")
    class_head = ClassificationHead(d_model, num_classes, pooling='attention')
    logits = class_head(x)
    print(f"  Logits shape: {logits.shape}")
    
    # Test MultiTaskHead
    print("\nTesting MultiTaskHead...")
    multi_head = MultiTaskHead(d_model, num_classes)
    output = multi_head(x, return_attention=True)
    print(f"  Boundary scores shape: {output['boundary_scores'].shape}")
    print(f"  Class logits shape: {output['class_logits'].shape}")
    
    # Test losses
    print("\nTesting losses...")
    boundary_loss_fn = BoundaryLoss(pos_weight=5.0)
    target = torch.randint(0, 2, (batch_size, seq_len)).float()
    loss = boundary_loss_fn(boundary_scores, target)
    print(f"  Boundary loss: {loss.item():.4f}")
    
    print("\nAll tests passed!")


# Alias for backward compatibility
TransitionalActivityModule = TransitionalActivityHead

