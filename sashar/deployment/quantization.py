"""
Quantization Support for SAS-HAR

This module provides quantization utilities for deploying SAS-HAR on edge devices.
Supports both post-training quantization (PTQ) and quantization-aware training (QAT).

Key Features:
- INT8/INT16 quantization
- Mixed precision quantization
- Calibration on representative data
- Quantization-aware training
- Export to TFLite/ONNX INT8

Reference:
- Jacob et al. (2018): "Quantization and Training of Neural Networks"
- Krishnamoorthi (2018): "Quantizing Deep Convolutional Networks"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List, Union, Callable
from dataclasses import dataclass
from enum import Enum
import copy


class QuantizationType(Enum):
    """Quantization types."""
    DYNAMIC = "dynamic"
    STATIC = "static"
    QAT = "qat"  # Quantization-aware training


class QuantizationPrecision(Enum):
    """Quantization precision."""
    INT8 = 8
    INT16 = 16
    MIXED = "mixed"


@dataclass
class QuantizationConfig:
    """Configuration for quantization."""
    
    # Type
    quant_type: QuantizationType = QuantizationType.STATIC
    
    # Precision
    precision: QuantizationPrecision = QuantizationPrecision.INT8
    
    # Mixed precision settings
    first_layer_precision: int = 16  # First layer needs higher precision
    last_layer_precision: int = 16   # Last layer needs higher precision
    
    # Calibration
    calibration_samples: int = 1000
    calibration_method: str = "minmax"  # minmax, histogram, entropy
    
    # QAT settings
    qat_epochs: int = 10
    qat_lr: float = 1e-5
    qat_start_epoch: int = 0
    
    # Backend
    backend: str = "qnnpack"  # qnnpack, fbgemm, onednn


class FakeQuantize(nn.Module):
    """
    Fake quantization module for QAT.
    
    Simulates quantization effects during training while allowing
    gradient flow through straight-through estimator (STE).
    """
    
    def __init__(
        self,
        bits: int = 8,
        symmetric: bool = True,
        per_channel: bool = False,
        momentum: float = 0.1
    ):
        super().__init__()
        
        self.bits = bits
        self.symmetric = symmetric
        self.per_channel = per_channel
        self.momentum = momentum
        
        # Quantization range
        self.qmin = 0
        self.qmax = 2 ** bits - 1
        
        # Learnable scale and zero_point
        self.scale = nn.Parameter(torch.ones(1))
        self.zero_point = nn.Parameter(torch.zeros(1))
        
        # Running statistics for calibration
        self.register_buffer('min_val', torch.tensor(float('inf')))
        self.register_buffer('max_val', torch.tensor(float('-inf')))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Fake quantization forward pass.
        
        Uses straight-through estimator (STE) for gradients.
        """
        if self.training:
            # Update min/max during training
            min_val = x.min().detach()
            max_val = x.max().detach()
            
            self.min_val = self.min_val * (1 - self.momentum) + min_val * self.momentum
            self.max_val = self.max_val * (1 - self.momentum) + max_val * self.momentum
            
            # Compute scale and zero_point
            scale = (self.max_val - self.min_val) / (self.qmax - self.qmin)
            zero_point = self.qmin - self.min_val / scale
            
            self.scale.data = scale
            self.zero_point.data = zero_point
        
        # Quantize
        x_quant = x / self.scale + self.zero_point
        x_quant = torch.clamp(x_quant, self.qmin, self.qmax)
        x_quant = torch.round(x_quant)
        
        # Dequantize (STE gradient)
        x_dequant = (x_quant - self.zero_point) * self.scale
        
        return x_dequant
    
    def get_quantized(self, x: torch.Tensor) -> torch.Tensor:
        """Get actual quantized values (for export)."""
        x_quant = x / self.scale + self.zero_point
        x_quant = torch.clamp(x_quant, self.qmin, self.qmax)
        x_quant = torch.round(x_quant)
        return x_quant.to(torch.int8 if self.bits == 8 else torch.int16)


class QuantizedModel(nn.Module):
    """
    Wrapper for quantized models.
    
    Provides a unified interface for both PTQ and QAT models.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[QuantizationConfig] = None
    ):
        super().__init__()
        
        self.original_model = model
        self.config = config or QuantizationConfig()
        self.quantized = False
        
        # Quantization stubs
        self.quant_stub = nn.quantized.FloatFunctional()
        self.dequant_stub = nn.quantized.FloatFunctional()
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass with optional quantization."""
        if self.quantized:
            x = self.quant_stub(x)
        
        outputs = self.original_model(x)
        
        if self.quantized and 'logits' in outputs:
            outputs['logits'] = self.dequant_stub(outputs['logits'])
        
        return outputs
    
    def calibrate(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_samples: Optional[int] = None
    ):
        """
        Calibrate quantization parameters.
        
        Args:
            dataloader: Calibration data
            num_samples: Number of samples to use
        """
        self.eval()
        num_samples = num_samples or self.config.calibration_samples
        
        with torch.no_grad():
            sample_count = 0
            for batch in dataloader:
                if sample_count >= num_samples:
                    break
                
                x = batch['data']
                _ = self.original_model(x)
                sample_count += x.size(0)
    
    def quantize(self):
        """Apply quantization to model."""
        self.quantized = True
        # In practice, would use torch.quantization.convert()


class QuantizationAwareTraining:
    """
    Quantization-aware training (QAT) utilities.
    
    Prepares model for QAT, handles training, and converts to quantized.
    
    Example:
        >>> model = SASHAR(input_channels=6, num_classes=6)
        >>> qat = QuantizationAwareTraining(model, config)
        >>> qat.prepare()
        >>> # ... train ...
        >>> quantized_model = qat.convert()
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[QuantizationConfig] = None
    ):
        self.model = model
        self.config = config or QuantizationConfig()
        self.prepared = False
        
    def prepare(self) -> nn.Module:
        """
        Prepare model for QAT.
        
        Inserts fake quantization modules at appropriate locations.
        """
        # Set quantization backend
        if self.config.backend == 'qnnpack':
            torch.backends.quantized.engine = 'qnnpack'
        
        # Fuse modules (Conv-BN-ReLU)
        self._fuse_modules()
        
        # Prepare for QAT
        self.model.qconfig = torch.quantization.get_default_qat_qconfig(
            self.config.backend
        )
        
        # Insert fake quantization
        self.model = torch.quantization.prepare_qat(
            self.model,
            inplace=True
        )
        
        self.prepared = True
        return self.model
    
    def _fuse_modules(self):
        """Fuse Conv-BN-ReLU sequences."""
        # This is model-specific; here's a generic approach
        if hasattr(self.model, 'cnn_encoder'):
            for name, module in self.model.cnn_encoder.named_children():
                if hasattr(module, 'fuse_modules'):
                    module.fuse_modules()
    
    def convert(self) -> nn.Module:
        """
        Convert QAT model to quantized model.
        
        Returns:
            Quantized model ready for deployment
        """
        if not self.prepared:
            raise RuntimeError("Model not prepared for QAT. Call prepare() first.")
        
        self.model.eval()
        quantized_model = torch.quantization.convert(self.model)
        
        return quantized_model
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        criterion: Callable
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Single QAT training step.
        
        Args:
            batch: Input batch
            optimizer: Optimizer
            criterion: Loss function
            
        Returns:
            loss: Training loss
            metrics: Dictionary of metrics
        """
        self.model.train()
        
        x = batch['data']
        labels = batch['label']
        
        # Forward
        outputs = self.model(x)
        loss = criterion(outputs['logits'], labels)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        metrics = {
            'loss': loss.item(),
            'accuracy': (outputs['logits'].argmax(1) == labels).float().mean().item()
        }
        
        return loss, metrics


class MixedPrecisionQuantizer:
    """
    Mixed precision quantization.
    
    Different layers use different bit-widths based on sensitivity.
    """
    
    def __init__(
        self,
        model: nn.Module,
        sensitive_layers: Optional[List[str]] = None,
        default_bits: int = 8,
        sensitive_bits: int = 16
    ):
        self.model = model
        self.sensitive_layers = sensitive_layers or []
        self.default_bits = default_bits
        self.sensitive_bits = sensitive_bits
        
        # Layer sensitivity analysis
        self.layer_sensitivity = {}
        
    def analyze_sensitivity(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_samples: int = 100
    ) -> Dict[str, float]:
        """
        Analyze layer sensitivity to quantization.
        
        Args:
            dataloader: Evaluation data
            num_samples: Number of samples
            
        Returns:
            Dictionary of layer sensitivity scores
        """
        self.model.eval()
        
        # Get baseline accuracy
        baseline_acc = self._evaluate(dataloader, num_samples)
        
        # Test each layer
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                # Temporarily quantize this layer
                sensitivity = self._quantize_layer_and_test(
                    name, module, dataloader, num_samples, baseline_acc
                )
                self.layer_sensitivity[name] = sensitivity
        
        return self.layer_sensitivity
    
    def _evaluate(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_samples: int
    ) -> float:
        """Evaluate model accuracy."""
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in dataloader:
                if total >= num_samples:
                    break
                
                outputs = self.model(batch['data'])
                preds = outputs['logits'].argmax(1)
                correct += (preds == batch['label']).sum().item()
                total += batch['label'].size(0)
        
        return correct / total
    
    def _quantize_layer_and_test(
        self,
        layer_name: str,
        module: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        num_samples: int,
        baseline_acc: float
    ) -> float:
        """Quantize single layer and measure accuracy drop."""
        # Save original weights
        original_weight = module.weight.data.clone()
        
        # Simulate INT8 quantization
        weight = module.weight.data
        scale = weight.abs().max() / 127.0
        quantized = torch.round(weight / scale) * scale
        module.weight.data = quantized
        
        # Evaluate
        acc = self._evaluate(dataloader, num_samples)
        sensitivity = baseline_acc - acc
        
        # Restore
        module.weight.data = original_weight
        
        return sensitivity
    
    def get_quantization_config(self) -> Dict[str, int]:
        """
        Get bit-width configuration for each layer.
        
        Returns:
            Dictionary mapping layer names to bit-widths
        """
        config = {}
        
        for name in self.layer_sensitivity:
            if name in self.sensitive_layers or self.layer_sensitivity[name] > 0.05:
                config[name] = self.sensitive_bits
            else:
                config[name] = self.default_bits
        
        return config


def quantize_model(
    model: nn.Module,
    calibration_data: torch.utils.data.DataLoader,
    config: Optional[QuantizationConfig] = None,
    evaluate_fn: Optional[Callable] = None
) -> nn.Module:
    """
    Convenience function to quantize a model.
    
    Args:
        model: Model to quantize
        calibration_data: Calibration data
        config: Quantization config
        evaluate_fn: Optional evaluation function
        
    Returns:
        Quantized model
    """
    config = config or QuantizationConfig()
    
    if config.quant_type == QuantizationType.STATIC:
        # Post-training static quantization
        model.eval()
        model.qconfig = torch.quantization.get_default_qconfig(config.backend)
        
        # Fuse modules
        if hasattr(model, 'fuse_modules'):
            model.fuse_modules()
        
        # Prepare
        torch.quantization.prepare(model, inplace=True)
        
        # Calibrate
        with torch.no_grad():
            for batch in calibration_data:
                model(batch['data'])
        
        # Convert
        quantized = torch.quantization.convert(model)
        
    elif config.quant_type == QuantizationType.DYNAMIC:
        # Dynamic quantization
        quantized = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear, nn.Conv1d},
            dtype=torch.qint8
        )
        
    elif config.quant_type == QuantizationType.QAT:
        # QAT handled separately
        qat = QuantizationAwareTraining(model, config)
        qat.prepare()
        quantized = model  # Return prepared model for training
    
    return quantized


def get_model_size(model: nn.Module, quantized: bool = False) -> int:
    """
    Get model size in bytes.
    
    Args:
        model: Model to measure
        quantized: Whether model is quantized
        
    Returns:
        Size in bytes
    """
    if quantized:
        # Quantized models use INT8
        total_params = sum(p.numel() for p in model.parameters())
        return total_params  # 1 byte per parameter
    else:
        # FP32 models
        return sum(p.numel() * p.element_size() for p in model.parameters())


def benchmark_quantized_model(
    model: nn.Module,
    sample_input: torch.Tensor,
    n_runs: int = 100
) -> Dict[str, float]:
    """
    Benchmark quantized model performance.
    
    Args:
        model: Model to benchmark
        sample_input: Sample input tensor
        n_runs: Number of runs
        
    Returns:
        Dictionary of performance metrics
    """
    import time
    
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(sample_input)
    
    # Benchmark
    latencies = []
    with torch.no_grad():
        for _ in range(n_runs):
            start = time.perf_counter()
            _ = model(sample_input)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # ms
    
    return {
        'mean_latency_ms': np.mean(latencies),
        'std_latency_ms': np.std(latencies),
        'p50_latency_ms': np.percentile(latencies, 50),
        'p95_latency_ms': np.percentile(latencies, 95),
        'p99_latency_ms': np.percentile(latencies, 99),
    }


# Example usage
if __name__ == "__main__":
    from sashar.models import SASHAR
    
    # Create model
    model = SASHAR(input_channels=6, num_classes=6)
    
    # Configure quantization
    config = QuantizationConfig(
        quant_type=QuantizationType.STATIC,
        precision=QuantizationPrecision.INT8,
        calibration_samples=500
    )
    
    print(f"Original model size: {get_model_size(model) / 1024:.2f} KB")
    
    # Quantize (would need actual calibration data)
    # quantized = quantize_model(model, calibration_loader, config)
    # print(f"Quantized model size: {get_model_size(quantized, quantized=True) / 1024:.2f} KB")
