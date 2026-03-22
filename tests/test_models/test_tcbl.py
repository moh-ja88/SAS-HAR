"""
Test suite for TCBL (Temporal Contrastive Boundary Learning) module.

Tests cover:
- Contrastive loss computations
- Boundary contrastive loss
- Temporal consistency loss
- Data augmentation
- Pseudo-label generation
"""

import pytest
import torch
import torch.nn as nn
import numpy as np

from sashar.models.tcbl import (
    TemporalContrastiveLoss,
    BoundaryContrastiveLoss,
    TemporalConsistencyLoss,
    TCBLPretrainer,
    ActivityAugmentation,
    PseudoLabelGenerator
)


class TestTemporalContrastiveLoss:
    """Test suite for temporal contrastive loss."""
    
    @pytest.fixture
    def loss_fn(self):
        """Create temporal contrastive loss function."""
        return TemporalContrastiveLoss(
            temperature=0.1,
            boundary_weight=2.0
        )
    
    @pytest.fixture
    def sample_embeddings(self):
        """Create sample embeddings for testing."""
        batch_size = 4
        seq_len = 32
        hidden_dim = 128
        
        z1 = torch.randn(batch_size, seq_len, hidden_dim)
        z2 = torch.randn(batch_size, seq_len, hidden_dim)
        
        return z1, z2
    
    def test_loss_computation(self, loss_fn, sample_embeddings):
        """Test loss computes without error."""
        z1, z2 = sample_embeddings
        
        loss, info = loss_fn(z1, z2)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar
        assert loss.item() >= 0  # Non-negative
        assert not torch.isnan(loss)
    
    def test_loss_with_boundary_mask(self, loss_fn, sample_embeddings):
        """Test loss with boundary mask."""
        z1, z2 = sample_embeddings
        batch_size, seq_len, _ = z1.shape
        
        boundary_mask = torch.zeros(batch_size, seq_len)
        boundary_mask[:, 10] = 1.0
        boundary_mask[:, 20] = 1.0
        
        loss, info = loss_fn(z1, z2, boundary_mask=boundary_mask)
        
        assert not torch.isnan(loss)
        assert 'contrastive_loss' in info
    
    def test_loss_with_activity_labels(self, loss_fn, sample_embeddings):
        """Test loss with activity labels."""
        z1, z2 = sample_embeddings
        batch_size, seq_len, _ = z1.shape
        
        activity_labels = torch.randint(0, 6, (batch_size, seq_len))
        
        loss, info = loss_fn(z1, z2, activity_labels=activity_labels)
        
        assert not torch.isnan(loss)
    
    def test_gradient_flow(self, loss_fn, sample_embeddings):
        """Test gradients flow through loss."""
        z1, z2 = sample_embeddings
        z1.requires_grad_(True)
        z2.requires_grad_(True)
        
        loss, _ = loss_fn(z1, z2)
        loss.backward()
        
        assert z1.grad is not None
        assert z2.grad is not None
        assert not torch.isnan(z1.grad).any()
    
    def test_temperature_effect(self, sample_embeddings):
        """Test different temperatures produce different losses."""
        z1, z2 = sample_embeddings
        
        loss_low_temp = TemporalContrastiveLoss(temperature=0.05)
        loss_high_temp = TemporalContrastiveLoss(temperature=0.5)
        
        loss1, _ = loss_low_temp(z1, z2)
        loss2, _ = loss_high_temp(z1, z2)
        
        # Different temperatures should give different losses
        assert not torch.allclose(loss1, loss2)
    
    def test_identical_embeddings(self, loss_fn):
        """Test loss with identical embeddings."""
        z = torch.randn(2, 16, 64)
        
        loss, info = loss_fn(z, z)
        
        # Identical embeddings should have low loss
        assert loss.item() < 5.0  # Reasonable bound


class TestBoundaryContrastiveLoss:
    """Test suite for boundary contrastive loss."""
    
    @pytest.fixture
    def loss_fn(self):
        """Create boundary contrastive loss function."""
        return BoundaryContrastiveLoss(temperature=0.1, margin=0.5)
    
    def test_loss_computation(self, loss_fn):
        """Test loss computes correctly."""
        embeddings = torch.randn(4, 32, 64)
        boundaries = torch.zeros(4, 32)
        boundaries[:, 10] = 1.0
        boundaries[:, 20] = 1.0
        
        loss, info = loss_fn(embeddings, boundaries)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert not torch.isnan(loss)
    
    def test_no_boundaries(self, loss_fn):
        """Test loss when no boundaries present."""
        embeddings = torch.randn(2, 16, 64)
        boundaries = torch.zeros(2, 16)  # No boundaries
        
        loss, info = loss_fn(embeddings, boundaries)
        
        # Should handle gracefully
        assert not torch.isnan(loss) or loss.item() == 0.0
    
    def test_all_boundaries(self, loss_fn):
        """Test loss when everything is boundary."""
        embeddings = torch.randn(2, 16, 64)
        boundaries = torch.ones(2, 16)  # All boundaries
        
        loss, info = loss_fn(embeddings, boundaries)
        
        # Should handle gracefully
        assert not torch.isnan(loss) or loss.item() == 0.0


class TestTemporalConsistencyLoss:
    """Test suite for temporal consistency loss."""
    
    @pytest.fixture
    def loss_fn(self):
        """Create temporal consistency loss function."""
        return TemporalConsistencyLoss(
            smoothness_weight=1.0,
            sharpness_weight=1.0
        )
    
    def test_loss_computation(self, loss_fn):
        """Test loss computes correctly."""
        predictions = torch.randn(4, 32, 64)
        boundaries = torch.zeros(4, 32)
        boundaries[:, 15] = 1.0
        
        activities = torch.zeros(4, 32, dtype=torch.long)
        activities[:, 16:] = 1
        
        loss, info = loss_fn(predictions, boundaries, activities)
        
        assert isinstance(loss, torch.Tensor)
        assert 'smoothness_loss' in info
        assert 'sharpness_loss' in info
    
    def test_smooth_predictions_low_loss(self, loss_fn):
        """Test smooth predictions in non-boundary regions."""
        # Smooth predictions (small differences)
        base = torch.randn(1, 1, 64)
        predictions = base.repeat(1, 32, 1)
        predictions[:, 1:, :] += torch.randn(1, 31, 64) * 0.01  # Small noise
        
        boundaries = torch.zeros(1, 32)
        activities = torch.zeros(1, 32, dtype=torch.long)
        
        loss, info = loss_fn(predictions, boundaries, activities)
        
        # Smooth predictions should have low smoothness loss
        assert info['smoothness_loss'] < 0.1


class TestTCBLPretrainer:
    """Test suite for complete TCBL pretrainer."""
    
    @pytest.fixture
    def pretrainer(self):
        """Create TCBL pretrainer."""
        return TCBLPretrainer(
            d_model=128,
            projection_dim=32,
            temperature=0.1
        )
    
    @pytest.fixture
    def sample_features(self):
        """Create sample features."""
        return torch.randn(4, 32, 128)
    
    def test_pretrainer_forward(self, pretrainer, sample_features):
        """Test pretrainer forward pass."""
        augmented = torch.randn_like(sample_features)
        
        loss, info = pretrainer(sample_features, augmented)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert 'total_loss' in info
        assert 'contrastive_loss' in info
    
    def test_pretrainer_with_pseudo_labels(self, pretrainer, sample_features):
        """Test pretrainer with pseudo labels."""
        augmented = torch.randn_like(sample_features)
        
        boundary_pseudo = torch.zeros(4, 32)
        boundary_pseudo[:, 10] = 1.0
        
        activity_pseudo = torch.randint(0, 6, (4, 32))
        
        loss, info = pretrainer(
            sample_features,
            augmented,
            boundary_pseudo_labels=boundary_pseudo,
            activity_pseudo_labels=activity_pseudo
        )
        
        assert not torch.isnan(loss)
        assert 'boundary_loss' in info
    
    def test_pretrainer_gradient_flow(self, pretrainer, sample_features):
        """Test gradients flow through pretrainer."""
        augmented = torch.randn_like(sample_features)
        sample_features.requires_grad_(True)
        
        loss, _ = pretrainer(sample_features, augmented)
        loss.backward()
        
        assert sample_features.grad is not None


class TestActivityAugmentation:
    """Test suite for data augmentation."""
    
    @pytest.fixture
    def augmenter(self):
        """Create augmentation module."""
        return ActivityAugmentation(
            jitter_std=0.01,
            scale_range=(0.9, 1.1),
            rotation_prob=0.5
        )
    
    @pytest.fixture
    def sample_data(self):
        """Create sample sensor data."""
        return torch.randn(4, 32, 6)  # [B, T, C]
    
    def test_augmentation_shape(self, augmenter, sample_data):
        """Test augmentation preserves shape."""
        augmented = augmenter(sample_data)
        
        assert augmented.shape == sample_data.shape
    
    def test_jitter(self, augmenter, sample_data):
        """Test jitter augmentation."""
        jittered = augmenter.jitter(sample_data)
        
        assert jittered.shape == sample_data.shape
        assert not torch.allclose(jittered, sample_data)
    
    def test_scaling(self, augmenter, sample_data):
        """Test scaling augmentation."""
        scaled = augmenter.scaling(sample_data)
        
        assert scaled.shape == sample_data.shape
    
    def test_rotation(self, augmenter, sample_data):
        """Test rotation augmentation."""
        rotated = augmenter.rotation(sample_data)
        
        assert rotated.shape == sample_data.shape
    
    def test_augmentation_changes_data(self, augmenter, sample_data):
        """Test augmentation actually changes data."""
        augmented = augmenter(sample_data)
        
        # Augmented data should be different (probabilistically)
        assert not torch.allclose(augmented, sample_data, atol=1e-5)
    
    def test_deterministic_with_seed(self):
        """Test augmentation is deterministic with seed."""
        torch.manual_seed(42)
        augmenter = ActivityAugmentation()
        data = torch.randn(2, 16, 6)
        
        torch.manual_seed(42)
        aug1 = augmenter(data)
        torch.manual_seed(42)
        aug2 = augmenter(data)
        
        assert torch.allclose(aug1, aug2)


class TestPseudoLabelGenerator:
    """Test suite for pseudo-label generation."""
    
    @pytest.fixture
    def generator(self):
        """Create pseudo-label generator."""
        return PseudoLabelGenerator(
            boundary_threshold=0.3,
            min_segment_length=10
        )
    
    @pytest.fixture
    def sample_features(self):
        """Create sample features with clear transitions."""
        features = torch.randn(2, 64, 128)
        
        # Create clear transition at t=32
        features[0, :32, :] = torch.randn(32, 128) + 1.0
        features[0, 32:, :] = torch.randn(32, 128) - 1.0
        
        return features
    
    def test_boundary_pseudo_generation(self, generator, sample_features):
        """Test boundary pseudo-label generation."""
        boundaries = generator.generate_boundary_pseudo_labels(sample_features)
        
        assert boundaries.shape == sample_features.shape[:2]
        assert boundaries.dtype == torch.float32
        assert boundaries.max() <= 1.0
        assert boundaries.min() >= 0.0
    
    def test_activity_pseudo_generation(self, generator, sample_features):
        """Test activity pseudo-label generation."""
        activities = generator.generate_activity_pseudo_labels(
            sample_features, 
            num_clusters=6
        )
        
        assert activities.shape == sample_features.shape[:2]
        assert activities.dtype == torch.long
        assert activities.min() >= 0
        assert activities.max() < 6
    
    def test_forward_returns_both(self, generator, sample_features):
        """Test forward returns both boundary and activity labels."""
        boundaries, activities = generator(sample_features)
        
        assert boundaries.shape == sample_features.shape[:2]
        assert activities.shape == sample_features.shape[:2]


class TestIntegration:
    """Integration tests for TCBL module."""
    
    def test_full_pretraining_pipeline(self):
        """Test complete pre-training pipeline."""
        # Setup
        batch_size = 2
        seq_len = 32
        d_model = 64
        
        pretrainer = TCBLPretrainer(d_model=d_model)
        augmenter = ActivityAugmentation()
        pseudo_gen = PseudoLabelGenerator()
        
        # Create features
        features = torch.randn(batch_size, seq_len, d_model)
        
        # Augment
        augmented = augmenter(features)
        
        # Generate pseudo labels
        boundary_pseudo, activity_pseudo = pseudo_gen(features)
        
        # Pre-train step
        loss, info = pretrainer(
            features,
            augmented,
            boundary_pseudo_labels=boundary_pseudo,
            activity_pseudo_labels=activity_pseudo
        )
        
        # Verify
        assert not torch.isnan(loss)
        assert loss.item() >= 0
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        for param in pretrainer.parameters():
            if param.requires_grad:
                assert param.grad is not None or param.grad == 0
    
    def test_pretrainer_with_encoder(self):
        """Test pretrainer with actual encoder output."""
        from sashar.models.encoder import CNNFeatureEncoder
        
        # Setup
        encoder = CNNFeatureEncoder(input_channels=6, hidden_dim=128)
        pretrainer = TCBLPretrainer(d_model=128)
        augmenter = ActivityAugmentation()
        
        # Create sensor data
        sensor_data = torch.randn(2, 6, 128)  # [B, C, T]
        
        # Encode
        features = encoder(sensor_data)  # [B, hidden_dim]
        
        # For this test, create temporal features
        features = features.unsqueeze(1).repeat(1, 16, 1)  # [B, T, D]
        
        # Augment
        augmented = augmenter(features)
        
        # Pre-train
        loss, info = pretrainer(features, augmented)
        
        assert not torch.isnan(loss)


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_batch(self):
        """Test with empty batch."""
        loss_fn = TemporalContrastiveLoss()
        
        z = torch.randn(0, 16, 64)
        loss, _ = loss_fn(z, z)
        
        # Should handle gracefully
        assert not torch.isnan(loss)
    
    def test_single_timestep(self):
        """Test with single timestep."""
        loss_fn = TemporalContrastiveLoss()
        
        z = torch.randn(2, 1, 64)
        loss, _ = loss_fn(z, z)
        
        assert not torch.isnan(loss)
    
    def test_very_small_temperature(self):
        """Test with very small temperature."""
        loss_fn = TemporalContrastiveLoss(temperature=1e-6)
        
        z1 = torch.randn(2, 16, 64)
        z2 = torch.randn(2, 16, 64)
        
        loss, _ = loss_fn(z1, z2)
        
        # Should not explode
        assert loss.item() < 1000
    
    def test_large_batch(self):
        """Test with large batch size."""
        loss_fn = TemporalContrastiveLoss()
        
        z1 = torch.randn(64, 32, 64)
        z2 = torch.randn(64, 32, 64)
        
        loss, _ = loss_fn(z1, z2)
        
        assert not torch.isnan(loss)
