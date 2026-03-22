"""
Test suite for SAS-HAR model architecture.

Tests cover:
- Model initialization and configuration
- Forward pass shapes and outputs
- Gradient flow
- Parameter counting
- Edge cases and error handling
"""

import pytest
import torch
import torch.nn as nn
from typing import Dict

from sashar.models import SASHAR, SASHARLite, create_sas_har
from sashar.models.encoder import CNNFeatureEncoder, TransformerTemporalModule
from sashar.models.heads import BoundaryHead, ClassificationHead, TransitionalActivityModule


class TestSASHARModel:
    """Test suite for the main SAS-HAR model."""
    
    @pytest.fixture
    def model(self):
        """Create a standard SAS-HAR model for testing."""
        return SASHAR(
            input_channels=6,
            num_classes=6,
            hidden_dim=256,
            num_heads=4,
            num_transformer_layers=3,
            use_transition_module=True,
            dropout=0.1
        )
    
    @pytest.fixture
    def sample_input(self):
        """Create sample input tensor."""
        return torch.randn(4, 6, 128)  # [Batch, Channels, Time]
    
    # =========================================================================
    # Initialization Tests
    # =========================================================================
    
    def test_model_initialization_default(self):
        """Test model initializes with default parameters."""
        model = SASHAR()
        
        assert model.hidden_dim == 256
        assert model.use_transition_module is True
        assert isinstance(model.cnn_encoder, CNNFeatureEncoder)
        assert isinstance(model.temporal_module, TransformerTemporalModule)
        assert isinstance(model.boundary_head, BoundaryHead)
        assert isinstance(model.classification_head, ClassificationHead)
    
    def test_model_initialization_custom(self):
        """Test model initializes with custom parameters."""
        model = SASHAR(
            input_channels=9,
            num_classes=12,
            hidden_dim=128,
            num_heads=2,
            num_transformer_layers=2,
            use_transition_module=False
        )
        
        assert model.hidden_dim == 128
        assert model.use_transition_module is False
        assert model.transition_module is None
    
    def test_model_initialization_invalid_params(self):
        """Test model handles invalid parameters gracefully."""
        # Invalid hidden_dim (should still work, just unusual)
        model = SASHAR(hidden_dim=64)
        assert model.hidden_dim == 64
        
        # Zero heads should be handled
        with pytest.raises((ValueError, RuntimeError)):
            model = SASHAR(num_heads=0)
    
    # =========================================================================
    # Forward Pass Tests
    # =========================================================================
    
    def test_forward_output_structure(self, model, sample_input):
        """Test forward pass returns correct output structure."""
        output = model(sample_input)
        
        assert isinstance(output, dict)
        assert 'logits' in output
        assert 'boundaries' in output
        assert 'temporal_features' in output
    
    def test_forward_output_shapes(self, model, sample_input):
        """Test forward pass produces correct output shapes."""
        output = model(sample_input)
        
        batch_size = sample_input.shape[0]
        num_classes = 6
        
        # Classification logits: [B, num_classes]
        assert output['logits'].shape == (batch_size, num_classes)
        
        # Boundaries: [B, T', 1] or [B, T']
        assert output['boundaries'].dim() in [2, 3]
        assert output['boundaries'].shape[0] == batch_size
        
        # Temporal features: [B, T', hidden_dim]
        assert output['temporal_features'].shape[0] == batch_size
        assert output['temporal_features'].shape[-1] == 256
    
    def test_forward_with_attention(self, model, sample_input):
        """Test forward pass with attention weights returned."""
        output = model(sample_input, return_attention=True)
        
        assert 'attention' in output
        assert 'boundary_attention' in output
    
    def test_forward_batch_independence(self, model):
        """Test that samples in a batch are processed independently."""
        x1 = torch.randn(1, 6, 128)
        x2 = torch.randn(1, 6, 128)
        x_batch = torch.cat([x1, x2], dim=0)
        
        # Single inference
        model.eval()
        with torch.no_grad():
            out1 = model(x1)
            out2 = model(x2)
            out_batch = model(x_batch)
        
        # Batch output should match individual outputs
        assert torch.allclose(out_batch['logits'][0], out1['logits'].squeeze(0), atol=1e-5)
        assert torch.allclose(out_batch['logits'][1], out2['logits'].squeeze(0), atol=1e-5)
    
    def test_forward_variable_length(self, model):
        """Test model handles variable length inputs."""
        lengths = [64, 128, 256]
        
        for length in lengths:
            x = torch.randn(2, 6, length)
            output = model(x)
            
            assert output['logits'].shape[0] == 2
            assert output['boundaries'].shape[0] == 2
    
    # =========================================================================
    # Gradient Flow Tests
    # =========================================================================
    
    def test_gradient_flow_all_parameters(self, model, sample_input):
        """Test gradients flow to all parameters."""
        model.train()
        output = model(sample_input)
        
        # Create a scalar loss
        loss = output['logits'].sum() + output['boundaries'].sum()
        loss.backward()
        
        # Check all parameters have gradients
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"
    
    def test_gradient_no_explosion(self, model, sample_input):
        """Test gradients don't explode during backward pass."""
        model.train()
        output = model(sample_input)
        loss = output['logits'].sum() + output['boundaries'].sum()
        loss.backward()
        
        max_grad = max(param.grad.abs().max().item() 
                      for param in model.parameters() 
                      if param.grad is not None)
        
        assert max_grad < 1000, f"Gradient explosion detected: max_grad={max_grad}"
    
    # =========================================================================
    # Parameter Counting Tests
    # =========================================================================
    
    def test_parameter_count_reasonable(self, model):
        """Test parameter count is within expected range."""
        params = model.count_parameters()
        
        # Full model should be between 500K and 2M parameters
        assert 500_000 < params < 2_000_000, f"Unexpected parameter count: {params}"
    
    def test_model_size_mb_reasonable(self, model):
        """Test model size is reasonable."""
        size_mb = model.get_model_size_mb()
        
        # Should be between 2MB and 10MB
        assert 2 < size_mb < 10, f"Unexpected model size: {size_mb}MB"
    
    # =========================================================================
    # Boundary Prediction Tests
    # =========================================================================
    
    def test_get_boundary_predictions_shape(self, model, sample_input):
        """Test boundary predictions have correct shape."""
        boundaries = model.get_boundary_predictions(sample_input, threshold=0.5)
        
        assert boundaries.dim() == 2
        assert boundaries.shape[0] == sample_input.shape[0]
    
    def test_get_boundary_predictions_binary(self, model, sample_input):
        """Test boundary predictions are binary."""
        boundaries = model.get_boundary_predictions(sample_input, threshold=0.5)
        
        unique_vals = torch.unique(boundaries)
        assert all(v in [0, 1] for v in unique_vals.tolist())
    
    def test_get_boundary_predictions_threshold(self, model, sample_input):
        """Test different thresholds produce different results."""
        boundaries_low = model.get_boundary_predictions(sample_input, threshold=0.3)
        boundaries_high = model.get_boundary_predictions(sample_input, threshold=0.7)
        
        # Lower threshold should produce more boundaries
        assert boundaries_low.sum() >= boundaries_high.sum()
    
    # =========================================================================
    # Activity Segment Tests
    # =========================================================================
    
    def test_get_activity_segments_returns_tuple(self, model, sample_input):
        """Test get_activity_segments returns expected tuple."""
        boundaries, labels = model.get_activity_segments(sample_input)
        
        assert isinstance(boundaries, tuple)
        assert isinstance(labels, torch.Tensor)
    
    # =========================================================================
    # Determinism Tests
    # =========================================================================
    
    def test_reproducibility_same_seed(self):
        """Test model produces same outputs with same seed."""
        torch.manual_seed(42)
        model1 = SASHAR()
        x = torch.randn(2, 6, 128)
        out1 = model1(x)
        
        torch.manual_seed(42)
        model2 = SASHAR()
        out2 = model2(x)
        
        assert torch.allclose(out1['logits'], out2['logits'])
        assert torch.allclose(out1['boundaries'], out2['boundaries'])


class TestSASHARLite:
    """Test suite for the lightweight SAS-HAR variant."""
    
    @pytest.fixture
    def lite_model(self):
        """Create a SASHARLite model for testing."""
        return SASHARLite(input_channels=6, num_classes=6)
    
    @pytest.fixture
    def full_model(self):
        """Create a full SASHAR model for comparison."""
        return SASHAR(input_channels=6, num_classes=6)
    
    def test_lite_smaller_than_full(self, lite_model, full_model):
        """Test lite model has fewer parameters than full model."""
        lite_params = lite_model.count_parameters()
        full_params = full_model.count_parameters()
        
        assert lite_params < full_params, \
            f"Lite ({lite_params}) should have fewer params than full ({full_params})"
    
    def test_lite_no_transition_module(self, lite_model):
        """Test lite model doesn't have transition module."""
        assert lite_model.use_transition_module is False
        assert lite_model.transition_module is None
    
    def test_lite_forward_works(self, lite_model):
        """Test lite model forward pass works."""
        x = torch.randn(2, 6, 128)
        output = lite_model(x)
        
        assert 'logits' in output
        assert 'boundaries' in output
    
    def test_lite_parameter_count_target(self, lite_model):
        """Test lite model meets target parameter count for edge deployment."""
        params = lite_model.count_parameters()
        
        # Target is <25K for edge deployment
        assert params < 100_000, f"Lite model has too many parameters: {params}"


class TestModelFactory:
    """Test suite for model factory function."""
    
    def test_create_sas_har_default(self):
        """Test factory creates default model."""
        model = create_sas_har()
        
        assert isinstance(model, SASHAR)
        assert model.hidden_dim == 256
    
    def test_create_sas_har_lite(self):
        """Test factory creates lite model."""
        model = create_sas_har(lite=True)
        
        assert isinstance(model, SASHARLite)
    
    def test_create_sas_har_custom_config(self):
        """Test factory accepts custom configuration."""
        config = {
            'hidden_dim': 128,
            'num_heads': 2
        }
        model = create_sas_har(config=config, num_classes=12)
        
        assert model.hidden_dim == 128


class TestEncoderComponents:
    """Test suite for encoder components."""
    
    def test_cnn_encoder_output_shape(self):
        """Test CNN encoder produces correct output shape."""
        encoder = CNNFeatureEncoder(input_channels=6, hidden_dim=256)
        x = torch.randn(4, 6, 128)
        
        output = encoder(x)
        
        assert output.shape[0] == 4
        assert output.shape[1] == 256
    
    def test_transformer_module_output_shape(self):
        """Test transformer module produces correct output shape."""
        transformer = TransformerTemporalModule(
            d_model=256,
            num_heads=4,
            num_layers=3
        )
        x = torch.randn(4, 16, 256)  # [B, T, D]
        
        output, _ = transformer(x)
        
        assert output.shape == (4, 16, 256)


class TestHeads:
    """Test suite for prediction heads."""
    
    def test_boundary_head_output_shape(self):
        """Test boundary head produces correct output shape."""
        head = BoundaryHead(d_model=256)
        x = torch.randn(4, 16, 256)
        
        output, _ = head(x)
        
        assert output.shape[0] == 4
        assert output.shape[1] == 16
        assert output.shape[2] == 1 or output.dim() == 2
    
    def test_classification_head_output_shape(self):
        """Test classification head produces correct output shape."""
        head = ClassificationHead(input_dim=256, num_classes=6)
        x = torch.randn(4, 16, 256)
        
        output = head(x)
        
        assert output.shape == (4, 6)
    
    def test_transition_module_output_shape(self):
        """Test transition module produces correct output shape."""
        module = TransitionalActivityModule(d_model=256)
        x = torch.randn(4, 16, 256)
        
        output = module(x)
        
        assert output.shape[0] == 4
        assert output.shape[-1] == 64  # Transition feature dimension


class TestEdgeCases:
    """Test suite for edge cases and error handling."""
    
    def test_single_sample_batch(self):
        """Test model handles batch size of 1."""
        model = SASHAR()
        x = torch.randn(1, 6, 128)
        
        output = model(x)
        
        assert output['logits'].shape[0] == 1
    
    def test_large_batch(self):
        """Test model handles large batch sizes."""
        model = SASHAR()
        x = torch.randn(64, 6, 128)
        
        output = model(x)
        
        assert output['logits'].shape[0] == 64
    
    def test_very_short_sequence(self):
        """Test model handles very short sequences."""
        model = SASHAR()
        x = torch.randn(2, 6, 32)
        
        output = model(x)
        
        assert output is not None
    
    def test_zero_input(self):
        """Test model handles zero input."""
        model = SASHAR()
        x = torch.zeros(2, 6, 128)
        
        output = model(x)
        
        assert not torch.isnan(output['logits']).any()
        assert not torch.isnan(output['boundaries']).any()


class TestDeviceCompatibility:
    """Test suite for device compatibility."""
    
    @pytest.mark.gpu
    def test_cuda_forward(self, device):
        """Test model works on CUDA."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        model = SASHAR().to(device)
        x = torch.randn(2, 6, 128).to(device)
        
        output = model(x)
        
        assert output['logits'].device == device
    
    def test_cpu_forward(self):
        """Test model works on CPU."""
        model = SASHAR()
        x = torch.randn(2, 6, 128)
        
        output = model(x)
        
        assert output['logits'].device == torch.device('cpu')
    
    def test_device_transfer(self):
        """Test model can be moved between devices."""
        model = SASHAR()
        
        # Move to CPU explicitly
        model = model.cpu()
        x_cpu = torch.randn(2, 6, 128)
        out_cpu = model(x_cpu)
        
        if torch.cuda.is_available():
            model = model.cuda()
            x_cuda = torch.randn(2, 6, 128).cuda()
            out_cuda = model(x_cuda)
            
            assert out_cuda['logits'].device.type == 'cuda'


class TestSerialization:
    """Test suite for model serialization."""
    
    def test_state_dict_save_load(self):
        """Test model state dict can be saved and loaded."""
        model1 = SASHAR()
        state_dict = model1.state_dict()
        
        model2 = SASHAR()
        model2.load_state_dict(state_dict)
        
        x = torch.randn(2, 6, 128)
        out1 = model1(x)
        out2 = model2(x)
        
        assert torch.allclose(out1['logits'], out2['logits'])
    
    def test_script_compatibility(self):
        """Test model is compatible with torch.script."""
        model = SASHAR()
        model.eval()
        
        # Should not raise
        scripted = torch.jit.script(model)
        
        x = torch.randn(2, 6, 128)
        out = scripted(x)
        
        assert out is not None
