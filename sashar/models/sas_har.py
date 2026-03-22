"""
SAS-HAR: Main Model Implementation
Self-Supervised Attention-based Segmentation for Human Activity Recognition
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from .encoder import CNNFeatureEncoder, TransformerTemporalModule
from .heads import BoundaryHead, ClassificationHead, TransitionalActivityModule


class SASHAR(nn.Module):
    """
    SAS-HAR: Self-Supervised Attention-based Segmentation for HAR
    
    A unified framework combining:
    - CNN feature encoder for local spatial features
    - Transformer temporal module for long-range dependencies
    - Boundary head for segmentation
    - Transitional activity module for transition specialization
    - Classification head for activity recognition
    
    Args:
        input_channels: Number of sensor channels (default: 6 for acc+gyro)
        num_classes: Number of activity classes
        hidden_dim: Hidden dimension (default: 256)
        num_heads: Number of attention heads (default: 4)
        num_transformer_layers: Number of transformer layers (default: 3)
        use_transition_module: Whether to use TASM (default: True)
        dropout: Dropout rate (default: 0.1)
    
    Example:
        >>> model = SASHAR(input_channels=6, num_classes=6)
        >>> outputs = model(sensor_data)
        >>> boundaries = outputs['boundaries']
        >>> classes = outputs['logits']
    """
    
    def __init__(
        self,
        input_channels: int = 6,
        num_classes: int = 6,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_transformer_layers: int = 3,
        use_transition_module: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.use_transition_module = use_transition_module
        
        # Core encoder
        self.cnn_encoder = CNNFeatureEncoder(
            input_channels=input_channels,
            hidden_dims=[hidden_dim // 4, hidden_dim // 2, hidden_dim]
        )
        
        self.temporal_module = TransformerTemporalModule(
            dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_transformer_layers,
            dropout=dropout
        )
        
        # Task-specific heads
        self.boundary_head = BoundaryHead(
            d_model=hidden_dim,
            dropout=dropout
        )
        
        if use_transition_module:
            self.transition_module = TransitionalActivityModule(
                d_model=hidden_dim
            )
            classifier_input_dim = hidden_dim + 64  # 64 from TASM
        else:
            self.transition_module = None
            classifier_input_dim = hidden_dim
        
        self.classification_head = ClassificationHead(
            d_model=classifier_input_dim,
            num_classes=num_classes,
            dropout=dropout
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self, 
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input sensor data [Batch, Channels, Time]
            return_attention: Whether to return attention weights
        
        Returns:
            Dictionary containing:
                - 'logits': Classification logits [Batch, Num_Classes]
                - 'boundaries': Boundary probabilities [Batch, Time', 1]
                - 'attention': (Optional) Attention weights
        """
        # x: [B, C, T]
        batch_size = x.shape[0]
        
        # CNN encoding: [B, C, T] -> [B, hidden_dim, T']
        cnn_features = self.cnn_encoder(x)
        
        # Transpose for transformer: [B, T', hidden_dim]
        temporal_features = cnn_features.transpose(1, 2)
        
        # Transformer temporal modeling: [B, T', hidden_dim]
        temporal_features, attention_weights = self.temporal_module(
            temporal_features,
            return_attention=True
        )
        
        # Boundary detection: [B, T', 1]
        boundary_probs, boundary_attention = self.boundary_head(temporal_features)
        
        # Transitional activity features (if enabled)
        if self.use_transition_module:
            transition_output = self.transition_module(temporal_features)
            # Use transition scores as additional features
            transition_features = transition_output['transition_scores'].unsqueeze(-1)  # [B, T, 1]
            # Expand to 64 dims as expected by the classifier
            transition_features = transition_features.expand(-1, -1, 64)  # [B, T, 64]
            combined_features = torch.cat([temporal_features, transition_features], dim=-1)
        else:
            combined_features = temporal_features
        
        # Classification: [B, num_classes]
        logits = self.classification_head(combined_features)
        
        # Prepare outputs
        outputs = {
            'logits': logits,
            'boundaries': boundary_probs,
            'temporal_features': temporal_features,
        }
        
        if return_attention:
            outputs['attention'] = attention_weights
            outputs['boundary_attention'] = boundary_attention
        
        return outputs
    
    def get_boundary_predictions(
        self, 
        x: torch.Tensor, 
        threshold: float = 0.5
    ) -> torch.Tensor:
        """
        Get binary boundary predictions
        
        Args:
            x: Input sensor data [Batch, Channels, Time]
            threshold: Boundary threshold (default: 0.5)
        
        Returns:
            Binary boundary predictions [Batch, Time']
        """
        outputs = self.forward(x)
        boundary_probs = outputs['boundaries'].squeeze(-1)
        return (boundary_probs > threshold).long()
    
    def get_activity_segments(
        self,
        x: torch.Tensor,
        threshold: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get activity segments from continuous stream
        
        Args:
            x: Input sensor data [Batch, Channels, Time]
            threshold: Boundary threshold
        
        Returns:
            Tuple of (segment_boundaries, segment_labels)
        """
        outputs = self.forward(x)
        
        # Get boundaries
        boundary_probs = outputs['boundaries'].squeeze(-1)
        boundaries = (boundary_probs > threshold).nonzero(as_tuple=True)
        
        # Get activity labels
        logits = outputs['logits']
        labels = logits.argmax(dim=-1)
        
        return boundaries, labels
    
    def count_parameters(self) -> int:
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_size_mb(self) -> float:
        """Get model size in MB"""
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        return (param_size + buffer_size) / (1024 ** 2)


class SASHARLite(SASHAR):
    """
    Lightweight version of SAS-HAR for edge deployment
    
    Reduced parameters through:
    - Fewer transformer layers (2 instead of 3)
    - Smaller hidden dimension (128 instead of 256)
    - Optional transition module removal
    """
    
    def __init__(
        self,
        input_channels: int = 6,
        num_classes: int = 6,
        **kwargs
    ):
        super().__init__(
            input_channels=input_channels,
            num_classes=num_classes,
            hidden_dim=128,  # Reduced
            num_heads=2,     # Reduced
            num_transformer_layers=2,  # Reduced
            use_transition_module=False,  # Disabled for efficiency
            dropout=0.1,
            **kwargs
        )


def create_sas_har(
    config: Optional[Dict] = None,
    num_classes: int = 6,
    lite: bool = False
) -> SASHAR:
    """
    Factory function to create SAS-HAR model
    
    Args:
        config: Optional configuration dictionary
        num_classes: Number of activity classes
        lite: Whether to create lightweight version
    
    Returns:
        Configured SAS-HAR model
    """
    if config is None:
        config = {}
    
    default_config = {
        'input_channels': 6,
        'num_classes': num_classes,
        'hidden_dim': 128 if lite else 256,
        'num_heads': 2 if lite else 4,
        'num_transformer_layers': 2 if lite else 3,
        'use_transition_module': not lite,
        'dropout': 0.1,
    }
    
    # Override with provided config
    default_config.update(config)
    
    if lite:
        return SASHARLite(**default_config)
    else:
        return SASHAR(**default_config)


# Example usage and testing
if __name__ == "__main__":
    # Create model
    model = SASHAR(input_channels=6, num_classes=6)
    
    # Print model info
    print(f"SAS-HAR Model")
    print(f"Parameters: {model.count_parameters():,}")
    print(f"Size: {model.get_model_size_mb():.2f} MB")
    
    # Test forward pass
    batch_size = 4
    time_steps = 128
    x = torch.randn(batch_size, 6, time_steps)
    
    outputs = model(x, return_attention=True)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Boundaries shape: {outputs['boundaries'].shape}")
    
    # Test lite version
    model_lite = SASHARLite(input_channels=6, num_classes=6)
    print(f"\nLite Version")
    print(f"Parameters: {model_lite.count_parameters():,}")
    print(f"Size: {model_lite.get_model_size_mb():.2f} MB")
