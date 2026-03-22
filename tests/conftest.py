"""
Pytest configuration and fixtures for SAS-HAR test suite.

This module provides shared fixtures, configuration, and utilities
for testing the SAS-HAR framework.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import tempfile
import shutil


# ============================================================================
# Configuration
# ============================================================================

def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )
    config.addinivalue_line(
        "markers", "integration: marks integration tests"
    )


# ============================================================================
# Random Seed Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def random_seed():
    """Global random seed for reproducibility."""
    return 42


@pytest.fixture(scope="session", autouse=True)
def set_global_seed(random_seed):
    """Set random seeds for reproducibility across all tests."""
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)


# ============================================================================
# Device Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def device():
    """Get available device (GPU if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def cpu_device():
    """Force CPU device for tests that shouldn't use GPU."""
    return torch.device("cpu")


# ============================================================================
# Model Fixtures
# ============================================================================

@pytest.fixture
def model_config() -> Dict:
    """Default model configuration for testing."""
    return {
        'input_channels': 6,
        'num_classes': 6,
        'hidden_dim': 128,  # Smaller for faster tests
        'num_heads': 2,
        'num_transformer_layers': 2,
        'use_transition_module': True,
        'dropout': 0.1,
    }


@pytest.fixture
def sample_input() -> torch.Tensor:
    """Sample input tensor for model testing.
    
    Returns:
        Tensor of shape [batch_size, channels, time_steps]
    """
    return torch.randn(4, 6, 128)


@pytest.fixture
def sample_input_large() -> torch.Tensor:
    """Larger sample input for integration tests."""
    return torch.randn(16, 6, 256)


@pytest.fixture
def sample_boundaries() -> torch.Tensor:
    """Sample boundary labels for testing.
    
    Returns:
        Binary tensor of shape [batch_size, time_steps]
    """
    boundaries = torch.zeros(4, 128)
    # Add some boundaries at random positions
    boundaries[0, 32] = 1.0
    boundaries[0, 64] = 1.0
    boundaries[0, 96] = 1.0
    boundaries[1, 40] = 1.0
    boundaries[1, 80] = 1.0
    boundaries[2, 50] = 1.0
    boundaries[3, 30] = 1.0
    boundaries[3, 60] = 1.0
    boundaries[3, 90] = 1.0
    return boundaries


@pytest.fixture
def sample_labels() -> torch.Tensor:
    """Sample activity labels for testing.
    
    Returns:
        Tensor of shape [batch_size] with class indices
    """
    return torch.tensor([0, 1, 2, 3])


# ============================================================================
# Dataset Fixtures
# ============================================================================

@pytest.fixture
def synthetic_har_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic HAR data for testing.
    
    Returns:
        Tuple of (data, labels, boundaries)
        - data: [num_samples, channels, time_steps]
        - labels: [num_samples]
        - boundaries: [num_samples, time_steps]
    """
    np.random.seed(42)
    
    num_samples = 100
    num_channels = 6
    time_steps = 128
    num_classes = 6
    
    # Generate random sensor data
    data = np.random.randn(num_samples, num_channels, time_steps).astype(np.float32)
    
    # Generate random labels
    labels = np.random.randint(0, num_classes, num_samples)
    
    # Generate random boundaries (sparse)
    boundaries = np.zeros((num_samples, time_steps), dtype=np.float32)
    for i in range(num_samples):
        # Add 1-3 random boundaries per sample
        num_boundaries = np.random.randint(1, 4)
        positions = np.random.choice(range(10, time_steps - 10), num_boundaries, replace=False)
        boundaries[i, positions] = 1.0
    
    return data, labels, boundaries


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test data.
    
    Yields:
        Path to temporary directory (cleaned up after test)
    """
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


# ============================================================================
# Loss Function Fixtures
# ============================================================================

@pytest.fixture
def boundary_loss_config() -> Dict:
    """Configuration for boundary loss testing."""
    return {
        'focal_gamma': 2.0,
        'boundary_weight': 1.0,
        'pos_weight': 5.0,  # To handle class imbalance
    }


@pytest.fixture
def tcbl_config() -> Dict:
    """Configuration for TCBL pre-training testing."""
    return {
        'temperature': 0.1,
        'boundary_weight': 2.0,
        'consistency_weight': 0.5,
        'projection_dim': 64,
    }


# ============================================================================
# Evaluation Fixtures
# ============================================================================

@pytest.fixture
def predictions_and_targets() -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample predictions and targets for metric testing."""
    # Predictions (batch_size=8, num_classes=6)
    preds = torch.tensor([
        [0.8, 0.1, 0.05, 0.02, 0.02, 0.01],
        [0.1, 0.7, 0.1, 0.05, 0.03, 0.02],
        [0.05, 0.1, 0.75, 0.05, 0.03, 0.02],
        [0.02, 0.03, 0.05, 0.8, 0.05, 0.05],
        [0.01, 0.02, 0.03, 0.04, 0.85, 0.05],
        [0.1, 0.1, 0.1, 0.1, 0.1, 0.5],  # Hard case
        [0.7, 0.15, 0.05, 0.05, 0.03, 0.02],
        [0.05, 0.05, 0.05, 0.05, 0.05, 0.75],
    ])
    
    # Targets
    targets = torch.tensor([0, 1, 2, 3, 4, 5, 0, 5])
    
    return preds, targets


@pytest.fixture
def boundary_predictions_and_targets() -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample boundary predictions and targets for metric testing."""
    # Predictions (probabilities)
    preds = torch.zeros(1, 128)
    preds[0, 30:35] = 0.8  # Predicted boundary
    preds[0, 60:65] = 0.9  # Predicted boundary
    preds[0, 95:100] = 0.7  # Predicted boundary
    
    # Targets (binary)
    targets = torch.zeros(1, 128)
    targets[0, 32] = 1.0  # True boundary
    targets[0, 64] = 1.0  # True boundary
    targets[0, 96] = 1.0  # True boundary
    
    return preds, targets


# ============================================================================
# Utility Fixtures
# ============================================================================

@pytest.fixture
def assert_tensors_close():
    """Utility to assert tensors are close within tolerance."""
    def _assert(a: torch.Tensor, b: torch.Tensor, rtol: float = 1e-4, atol: float = 1e-6):
        assert torch.allclose(a, b, rtol=rtol, atol=atol), \
            f"Tensors not close:\nExpected: {a}\nGot: {b}"
    return _assert


@pytest.fixture
def count_parameters():
    """Utility to count model parameters."""
    def _count(model: torch.nn.Module) -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return _count


# ============================================================================
# Skip Markers
# ============================================================================

def pytest_collection_modifyitems(config, items):
    """Skip GPU tests if no GPU available."""
    skip_gpu = pytest.mark.skip(reason="GPU not available")
    
    for item in items:
        if "gpu" in item.keywords and not torch.cuda.is_available():
            item.add_marker(skip_gpu)
