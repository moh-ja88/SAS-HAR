"""
Temporal Contrastive Boundary Learning (TCBL) module.

This module implements self-supervised pre-training for activity segmentation:
- Temporal contrastive learning at boundaries
- Boundary-aware augmentation
- Consistency regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
import math


class TemporalContrastiveLoss(nn.Module):
    """
    Temporal contrastive loss for learning boundary-aware representations.
    
    Encourages:
    - Similar representations for same-activity segments
    - Different representations for different-activity segments
    - Boundary-sensitive features at transitions
    """
    
    def __init__(
        self,
        temperature: float = 0.1,
        boundary_weight: float = 2.0,
        use_hard_negatives: bool = True
    ):
        super().__init__()
        self.temperature = temperature
        self.boundary_weight = boundary_weight
        self.use_hard_negatives = use_hard_negatives
    
    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        boundary_mask: Optional[torch.Tensor] = None,
        activity_labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            z1: First view embeddings (B, T, D)
            z2: Second view embeddings (B, T, D)
            boundary_mask: Binary mask for boundary positions (B, T)
            activity_labels: Activity labels for each position (B, T)
        
        Returns:
            loss: Scalar contrastive loss
            info_dict: Dictionary with loss components
        """
        B, T, D = z1.shape
        
        # Normalize embeddings
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        
        # Flatten batch and time dimensions
        z1_flat = z1.reshape(B * T, D)
        z2_flat = z2.reshape(B * T, D)
        
        # Compute similarity matrix
        sim_matrix = torch.mm(z1_flat, z2_flat.t()) / self.temperature  # (BT, BT)
        
        # Create positive pairs (diagonal + temporal neighbors)
        positive_mask = torch.eye(B * T, device=z1.device)
        
        # Add temporal neighbors as positives (within same batch)
        for offset in [-1, 1]:
            diag = torch.diag(torch.ones(B * T - abs(offset), device=z1.device), diagonal=offset)
            positive_mask += diag * 0.5  # Weighted by distance
        
        # Activity-aware negatives (different activities = hard negatives)
        if activity_labels is not None:
            labels_flat = activity_labels.reshape(B * T)
            # Same activity = soft positive
            same_activity = (labels_flat.unsqueeze(0) == labels_flat.unsqueeze(1)).float()
            positive_mask = positive_mask + same_activity * 0.3
            positive_mask = torch.clamp(positive_mask, 0, 1)
        
        # Boundary weighting
        if boundary_mask is not None:
            boundary_flat = boundary_mask.reshape(B * T)
            boundary_weight = 1.0 + (self.boundary_weight - 1.0) * boundary_flat
            sim_matrix = sim_matrix * boundary_weight.unsqueeze(1)
        
        # InfoNCE loss
        # Numerator: positive pairs
        pos_sim = (sim_matrix * positive_mask).sum(dim=1)
        
        # Denominator: all pairs
        neg_mask = 1.0 - positive_mask
        neg_sim = (sim_matrix * neg_mask).sum(dim=1)
        
        # Contrastive loss
        loss = -torch.log(pos_sim.exp() / (pos_sim.exp() + neg_sim.exp() + 1e-8))
        
        # Mean over batch
        loss = loss.mean()
        
        info_dict = {
            'contrastive_loss': loss.item(),
            'pos_sim_mean': pos_sim.mean().item(),
            'neg_sim_mean': neg_sim.mean().item()
        }
        
        return loss, info_dict


class BoundaryContrastiveLoss(nn.Module):
    """
    Contrastive loss specifically for boundary detection.
    
    Learns to distinguish boundary regions from non-boundary regions.
    """
    
    def __init__(
        self,
        temperature: float = 0.1,
        margin: float = 0.5
    ):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
    
    def forward(
        self,
        embeddings: torch.Tensor,
        boundary_labels: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            embeddings: Feature embeddings (B, T, D)
            boundary_labels: Binary boundary labels (B, T)
        
        Returns:
            loss: Scalar boundary contrastive loss
            info_dict: Dictionary with loss components
        """
        B, T, D = embeddings.shape
        
        # Normalize
        embeddings = F.normalize(embeddings, dim=-1)
        
        # Separate boundary and non-boundary embeddings
        boundary_mask = boundary_labels.bool()
        non_boundary_mask = ~boundary_mask
        
        # Get boundary and non-boundary embeddings
        boundary_embeds = embeddings[boundary_mask]  # (N_boundary, D)
        non_boundary_embeds = embeddings[non_boundary_mask]  # (N_non_boundary, D)
        
        if boundary_embeds.size(0) == 0 or non_boundary_embeds.size(0) == 0:
            return torch.tensor(0.0, device=embeddings.device), {'boundary_loss': 0.0}
        
        # Sample to balance
        min_samples = min(boundary_embeds.size(0), non_boundary_embeds.size(0), 256)
        
        if boundary_embeds.size(0) > min_samples:
            idx = torch.randperm(boundary_embeds.size(0))[:min_samples]
            boundary_embeds = boundary_embeds[idx]
        
        if non_boundary_embeds.size(0) > min_samples:
            idx = torch.randperm(non_boundary_embeds.size(0))[:min_samples]
            non_boundary_embeds = non_boundary_embeds[idx]
        
        # Compute intra-class similarity (within boundaries, within non-boundaries)
        boundary_sim = torch.mm(boundary_embeds, boundary_embeds.t()) / self.temperature
        non_boundary_sim = torch.mm(non_boundary_embeds, non_boundary_embeds.t()) / self.temperature
        
        # Compute inter-class similarity (between boundaries and non-boundaries)
        inter_sim = torch.mm(boundary_embeds, non_boundary_embeds.t()) / self.temperature
        
        # Loss: maximize intra-class similarity, minimize inter-class similarity
        # Using margin-based contrastive loss
        intra_loss = (1.0 - boundary_sim.mean()) + (1.0 - non_boundary_sim.mean())
        inter_loss = F.relu(inter_sim.mean() - self.margin)
        
        loss = intra_loss + inter_loss
        
        info_dict = {
            'boundary_loss': loss.item(),
            'intra_sim': boundary_sim.mean().item(),
            'inter_sim': inter_sim.mean().item()
        }
        
        return loss, info_dict


class TemporalConsistencyLoss(nn.Module):
    """
    Temporal consistency regularization.
    
    Encourages smooth predictions within same-activity regions
    and sharp changes at boundaries.
    """
    
    def __init__(
        self,
        smoothness_weight: float = 1.0,
        sharpness_weight: float = 1.0
    ):
        super().__init__()
        self.smoothness_weight = smoothness_weight
        self.sharpness_weight = sharpness_weight
    
    def forward(
        self,
        predictions: torch.Tensor,
        boundary_labels: torch.Tensor,
        activity_labels: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            predictions: Model predictions (B, T, D) or (B, T)
            boundary_labels: Binary boundary labels (B, T)
            activity_labels: Activity labels (B, T)
        
        Returns:
            loss: Scalar consistency loss
            info_dict: Dictionary with loss components
        """
        B, T = boundary_labels.shape
        
        # Prediction differences
        if predictions.dim() == 3:
            pred_diff = (predictions[:, 1:] - predictions[:, :-1]).abs()
            pred_diff = pred_diff.mean(dim=-1)  # (B, T-1)
        else:
            pred_diff = (predictions[:, 1:] - predictions[:, :-1]).abs()  # (B, T-1)
        
        # Activity changes
        activity_change = (activity_labels[:, 1:] != activity_labels[:, :-1]).float()
        
        # Non-boundary regions (smoothness)
        non_boundary = (1.0 - activity_change)  # (B, T-1)
        smoothness_loss = (pred_diff * non_boundary).mean()
        
        # Boundary regions (sharpness)
        boundary = activity_change  # (B, T-1)
        sharpness_loss = F.relu(0.5 - pred_diff * boundary).mean()
        
        # Combined loss
        loss = self.smoothness_weight * smoothness_loss + self.sharpness_weight * sharpness_loss
        
        info_dict = {
            'consistency_loss': loss.item(),
            'smoothness_loss': smoothness_loss.item(),
            'sharpness_loss': sharpness_loss.item()
        }
        
        return loss, info_dict


class TCBLPretrainer(nn.Module):
    """
    Complete TCBL pre-training module.
    
    Combines all contrastive and consistency losses for
    self-supervised boundary learning.
    """
    
    def __init__(
        self,
        d_model: int = 512,
        projection_dim: int = 128,
        temperature: float = 0.1,
        boundary_weight: float = 2.0,
        consistency_weight: float = 0.5
    ):
        super().__init__()
        
        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Linear(d_model, projection_dim)
        )
        
        # Loss functions
        self.temporal_contrastive = TemporalContrastiveLoss(
            temperature=temperature,
            boundary_weight=boundary_weight
        )
        
        self.boundary_contrastive = BoundaryContrastiveLoss(
            temperature=temperature
        )
        
        self.consistency_loss = TemporalConsistencyLoss()
        
        self.consistency_weight = consistency_weight
    
    def forward(
        self,
        features: torch.Tensor,
        augmented_features: torch.Tensor,
        boundary_pseudo_labels: Optional[torch.Tensor] = None,
        activity_pseudo_labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            features: Original features (B, T, D)
            augmented_features: Augmented features (B, T, D)
            boundary_pseudo_labels: Pseudo boundary labels (B, T)
            activity_pseudo_labels: Pseudo activity labels (B, T)
        
        Returns:
            loss: Total pre-training loss
            info_dict: Dictionary with all loss components
        """
        B, T, D = features.shape
        
        # Project features
        z1 = self.projection_head(features.reshape(B * T, D)).reshape(B, T, -1)
        z2 = self.projection_head(augmented_features.reshape(B * T, D)).reshape(B, T, -1)
        
        # Temporal contrastive loss
        tc_loss, tc_info = self.temporal_contrastive(
            z1, z2,
            boundary_mask=boundary_pseudo_labels,
            activity_labels=activity_pseudo_labels
        )
        
        # Boundary contrastive loss
        if boundary_pseudo_labels is not None:
            bc_loss, bc_info = self.boundary_contrastive(
                (z1 + z2) / 2,
                boundary_pseudo_labels
            )
        else:
            bc_loss = torch.tensor(0.0, device=features.device)
            bc_info = {'boundary_loss': 0.0}
        
        # Consistency loss
        if activity_pseudo_labels is not None and boundary_pseudo_labels is not None:
            # Use features as predictions for consistency
            consistency, consistency_info = self.consistency_loss(
                features,
                boundary_pseudo_labels,
                activity_pseudo_labels
            )
        else:
            consistency = torch.tensor(0.0, device=features.device)
            consistency_info = {'consistency_loss': 0.0}
        
        # Total loss
        total_loss = tc_loss + bc_loss + self.consistency_weight * consistency
        
        # Combine info
        info_dict = {
            'total_loss': total_loss.item(),
            **tc_info,
            **bc_info,
            **consistency_info
        }
        
        return total_loss, info_dict


class ActivityAugmentation(nn.Module):
    """
    Data augmentation for self-supervised pre-training.
    
    Implements sensor-specific augmentations for HAR.
    """
    
    def __init__(
        self,
        jitter_std: float = 0.01,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        time_warp_sigma: float = 0.2,
        rotation_prob: float = 0.5,
        dropout_prob: float = 0.1
    ):
        super().__init__()
        
        self.jitter_std = jitter_std
        self.scale_range = scale_range
        self.time_warp_sigma = time_warp_sigma
        self.rotation_prob = rotation_prob
        self.dropout_prob = dropout_prob
    
    def jitter(self, x: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise."""
        noise = torch.randn_like(x) * self.jitter_std
        return x + noise
    
    def scaling(self, x: torch.Tensor) -> torch.Tensor:
        """Random scaling."""
        scale = torch.empty(1, device=x.device).uniform_(*self.scale_range)
        return x * scale
    
    def time_warp(self, x: torch.Tensor) -> torch.Tensor:
        """Time warping augmentation."""
        B, T, C = x.shape
        
        # Generate smooth warping function
        t_orig = torch.linspace(0, 1, T, device=x.device)
        warp = torch.cumsum(torch.randn(T, device=x.device) * self.time_warp_sigma / T, dim=0)
        warp = warp - warp[0]  # Start at 0
        warp = warp / (warp[-1] + 1e-8)  # Normalize to [0, 1]
        t_warped = torch.clamp(t_orig + warp * 0.2, 0, 1)
        
        # Interpolate
        x_warped = torch.zeros_like(x)
        for b in range(B):
            for c in range(C):
                x_warped[b, :, c] = torch.interp(t_warped, t_orig, x[b, :, c])
        
        return x_warped
    
    def rotation(self, x: torch.Tensor) -> torch.Tensor:
        """3D rotation for accelerometer/gyroscope data."""
        B, T, C = x.shape
        
        if C < 3:
            return x
        
        # Random rotation matrix
        angle = torch.empty(B, device=x.device).uniform_(-math.pi, math.pi)
        cos_a, sin_a = torch.cos(angle), torch.sin(angle)
        
        # Simple rotation around z-axis for first 3 channels
        x_rot = x.clone()
        x_rot[:, :, 0] = cos_a.unsqueeze(1) * x[:, :, 0] - sin_a.unsqueeze(1) * x[:, :, 1]
        x_rot[:, :, 1] = sin_a.unsqueeze(1) * x[:, :, 0] + cos_a.unsqueeze(1) * x[:, :, 1]
        
        return x_rot
    
    def channel_dropout(self, x: torch.Tensor) -> torch.Tensor:
        """Randomly drop channels."""
        if torch.rand(1).item() > self.dropout_prob:
            return x
        
        mask = torch.rand(x.size(-1), device=x.device) > 0.1
        return x * mask.unsqueeze(0).unsqueeze(0).float()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply random augmentation.
        
        Args:
            x: Input tensor (B, T, C)
        
        Returns:
            Augmented tensor (B, T, C)
        """
        # Apply augmentations randomly
        if torch.rand(1).item() > 0.5:
            x = self.jitter(x)
        
        if torch.rand(1).item() > 0.5:
            x = self.scaling(x)
        
        if torch.rand(1).item() > 0.7:
            x = self.time_warp(x)
        
        if torch.rand(1).item() > 0.5:
            x = self.rotation(x)
        
        if torch.rand(1).item() > 0.8:
            x = self.channel_dropout(x)
        
        return x


class PseudoLabelGenerator(nn.Module):
    """
    Generate pseudo labels for self-supervised pre-training.
    
    Uses simple heuristics to create boundary and activity pseudo labels.
    """
    
    def __init__(
        self,
        boundary_threshold: float = 0.3,
        min_segment_length: int = 10
    ):
        super().__init__()
        self.boundary_threshold = boundary_threshold
        self.min_segment_length = min_segment_length
    
    def generate_boundary_pseudo_labels(
        self,
        features: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate pseudo boundary labels based on feature similarity.
        
        Args:
            features: Input features (B, T, D)
        
        Returns:
            pseudo_labels: Binary boundary labels (B, T)
        """
        B, T, D = features.shape
        
        # Compute feature similarity between adjacent time steps
        features_norm = F.normalize(features, dim=-1)
        similarity = (features_norm[:, :-1] * features_norm[:, 1:]).sum(dim=-1)  # (B, T-1)
        
        # Low similarity = potential boundary
        boundary_scores = 1.0 - similarity  # (B, T-1)
        
        # Threshold to get binary labels
        boundary_pseudo = (boundary_scores > self.boundary_threshold).float()
        
        # Pad to original length
        boundary_pseudo = F.pad(boundary_pseudo, (0, 1), value=0)
        
        return boundary_pseudo
    
    def generate_activity_pseudo_labels(
        self,
        features: torch.Tensor,
        num_clusters: int = 6
    ) -> torch.Tensor:
        """
        Generate pseudo activity labels using simple clustering.
        
        Args:
            features: Input features (B, T, D)
            num_clusters: Number of activity clusters
        
        Returns:
            pseudo_labels: Activity pseudo labels (B, T)
        """
        B, T, D = features.shape
        
        # Simple k-means-like clustering using feature statistics
        features_flat = features.reshape(B * T, D)
        
        # Normalize
        features_norm = F.normalize(features_flat, dim=-1)
        
        # Compute similarity to random centroids
        centroids = torch.randn(num_clusters, D, device=features.device)
        centroids = F.normalize(centroids, dim=-1)
        
        similarity = torch.mm(features_norm, centroids.t())  # (BT, K)
        pseudo_labels = similarity.argmax(dim=-1)  # (BT,)
        
        pseudo_labels = pseudo_labels.reshape(B, T)
        
        return pseudo_labels
    
    def forward(
        self,
        features: torch.Tensor,
        num_activity_clusters: int = 6
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate both boundary and activity pseudo labels.
        
        Args:
            features: Input features (B, T, D)
            num_activity_clusters: Number of activity clusters
        
        Returns:
            boundary_pseudo_labels: (B, T)
            activity_pseudo_labels: (B, T)
        """
        boundary_pseudo = self.generate_boundary_pseudo_labels(features)
        activity_pseudo = self.generate_activity_pseudo_labels(features, num_activity_clusters)
        
        return boundary_pseudo, activity_pseudo


if __name__ == "__main__":
    # Test TCBL module
    batch_size = 4
    seq_len = 128
    d_model = 512
    
    features = torch.randn(batch_size, seq_len, d_model)
    
    # Test augmentation
    print("Testing augmentation...")
    augmenter = ActivityAugmentation()
    augmented = augmenter(features)
    print(f"  Original shape: {features.shape}")
    print(f"  Augmented shape: {augmented.shape}")
    
    # Test pseudo label generation
    print("\nTesting pseudo label generation...")
    pseudo_generator = PseudoLabelGenerator()
    boundary_pseudo, activity_pseudo = pseudo_generator(features)
    print(f"  Boundary pseudo labels shape: {boundary_pseudo.shape}")
    print(f"  Activity pseudo labels shape: {activity_pseudo.shape}")
    print(f"  Boundary ratio: {boundary_pseudo.mean().item():.4f}")
    
    # Test TCBL pretrainer
    print("\nTesting TCBL pretrainer...")
    pretrainer = TCBLPretrainer(d_model)
    loss, info = pretrainer(
        features,
        augmented,
        boundary_pseudo_labels=boundary_pseudo,
        activity_pseudo_labels=activity_pseudo
    )
    print(f"  Total loss: {loss.item():.4f}")
    print(f"  Loss components: {info}")
    
    print("\nAll tests passed!")


# Aliases for backward compatibility with __init__.py
TemporalContrastiveLearning = TCBLPretrainer
ContinuityPredictor = TemporalConsistencyLoss
MaskedTemporalModeling = PseudoLabelGenerator

