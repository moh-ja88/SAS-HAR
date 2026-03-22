"""
Model Export Utilities for HAR Deployment

Provides tools for exporting SAS-HAR models to production formats:
- ONNX (Open Neural Network Exchange)
- TFLite (TensorFlow Lite)
- TorchScript
- Quantized models
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray

try:
    import onnx
    import onnxruntime as ort
    _has_onnx = True
except ImportError:
    _has_onnx = False
    onnx = None  # type: ignore
    ort = None  # type: ignore

try:
    import tensorflow as tf  # type: ignore
    _has_tensorflow = True
except ImportError:
    _has_tensorflow = False
    tf = None  # type: ignore


class ModelExporter:
    """
    Export PyTorch models to various deployment formats.
    
    Supports:
    - ONNX format for cross-platform deployment
    - TorchScript for optimized PyTorch inference
    - TFLite for mobile/edge deployment (via ONNX-TF)
    - Quantized exports for reduced model size
    
    Example:
        >>> model = SASHARModel(...)
        >>> exporter = ModelExporter(model)
        >>> exporter.export_onnx('model.onnx', sample_input)
    """
    
    def __init__(
        self,
        model: nn.Module,
        input_shape: tuple[int, ...] = (1, 6, 128),
        dynamic_batch: bool = True,
    ):
        """
        Initialize model exporter.
        
        Args:
            model: PyTorch model to export
            input_shape: Input tensor shape (B, C, T) for tracing
            dynamic_batch: Whether to use dynamic batch dimension
        """
        self.model = model
        self.input_shape = input_shape
        self.dynamic_batch = dynamic_batch
        self.model.eval()
    
    def export_onnx(
        self,
        output_path: str | Path,
        sample_input: torch.Tensor | None = None,
        opset_version: int = 14,
        simplify: bool = True,
        check_model: bool = True,
        input_names: list[str] | None = None,
        output_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Export model to ONNX format.
        
        Args:
            output_path: Path to save ONNX model
            sample_input: Sample input for tracing (created if None)
            opset_version: ONNX opset version
            simplify: Whether to simplify the ONNX model
            check_model: Whether to validate the exported model
            input_names: Names for input tensors
            output_names: Names for output tensors
        
        Returns:
            Dictionary with export metadata
        """
        if not _has_onnx:
            raise ImportError(
                "ONNX export requires onnx and onnxruntime. "
                "Install with: pip install onnx onnxruntime"
            )
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if sample_input is None:
            sample_input = torch.randn(self.input_shape)
        
        if input_names is None:
            input_names = ['sensor_data']
        
        if output_names is None:
            output_names = ['class_logits', 'boundary_logits']
        
        # Dynamic axes for variable batch size
        dynamic_axes = None
        if self.dynamic_batch:
            dynamic_axes = {
                name: {0: 'batch_size'}
                for name in input_names + output_names
            }
        
        # Export
        with torch.no_grad():
            torch.onnx.export(
                self.model,
                sample_input,
                str(output_path),
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
            )
        
        # Check model
        if check_model:
            onnx_model = onnx.load(str(output_path))
            onnx.checker.check_model(onnx_model)
        
        # Simplify if requested
        if simplify:
            try:
                import onnxsim  # type: ignore
                onnxsim.simplify(str(output_path))
            except ImportError:
                pass  # Simplification is optional
        
        # Verify with ONNX Runtime
        ort_session = ort.InferenceSession(str(output_path))
        input_name = ort_session.get_inputs()[0].name
        
        # Test inference
        test_input = sample_input.numpy()
        ort_outputs = ort_session.run(None, {input_name: test_input})
        
        # Compare with PyTorch
        with torch.no_grad():
            torch_outputs = self.model(sample_input)
            if isinstance(torch_outputs, dict):
                torch_outputs = (torch_outputs['logits'], torch_outputs.get('boundaries', torch.zeros(1)))
            elif isinstance(torch_outputs, torch.Tensor):
                torch_outputs = (torch_outputs,)
        
        # Calculate error
        max_diff = 0.0
        for ort_out, torch_out in zip(ort_outputs, torch_outputs):
            diff = np.max(np.abs(ort_out - torch_out.numpy()))
            max_diff = max(max_diff, float(diff))
        
        # Get model size
        model_size_mb = output_path.stat().st_size / (1024 * 1024)
        
        return {
            'path': str(output_path),
            'size_mb': model_size_mb,
            'opset_version': opset_version,
            'input_names': input_names,
            'output_names': output_names,
            'max_diff': max_diff,
            'input_shape': list(self.input_shape),
            'dynamic_batch': self.dynamic_batch,
        }
    
    def export_torchscript(
        self,
        output_path: str | Path,
        sample_input: torch.Tensor | None = None,
        method: str = 'trace',
        optimize: bool = True,
    ) -> dict[str, Any]:
        """
        Export model to TorchScript format.
        
        Args:
            output_path: Path to save TorchScript model
            sample_input: Sample input for tracing
            method: 'trace' or 'script'
            optimize: Whether to apply optimizations
        
        Returns:
            Dictionary with export metadata
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if sample_input is None:
            sample_input = torch.randn(self.input_shape)
        
        with torch.no_grad():
            if method == 'trace':
                scripted_model = torch.jit.trace(self.model, sample_input)
            else:
                scripted_model = torch.jit.script(self.model)
        
        if optimize:
            scripted_model = torch.jit.optimize_for_inference(scripted_model)
        
        scripted_model.save(str(output_path))
        
        # Verify
        loaded_model = torch.jit.load(str(output_path))
        with torch.no_grad():
            orig_output = self.model(sample_input)
            scripted_output = loaded_model(sample_input)
        
        # Calculate error
        if isinstance(orig_output, dict):
            orig_output = orig_output['logits']
        if isinstance(scripted_output, dict):
            scripted_output = scripted_output['logits']
        
        max_diff = float(torch.max(torch.abs(orig_output - scripted_output)))
        
        model_size_mb = output_path.stat().st_size / (1024 * 1024)
        
        return {
            'path': str(output_path),
            'size_mb': model_size_mb,
            'method': method,
            'optimized': optimize,
            'max_diff': max_diff,
        }
    
    def export_tflite(
        self,
        output_path: str | Path,
        sample_input: torch.Tensor | None = None,
        quantize: bool = False,
        representative_dataset: Any | None = None,
    ) -> dict[str, Any]:
        """
        Export model to TFLite format via ONNX.
        
        Args:
            output_path: Path to save TFLite model
            sample_input: Sample input for tracing
            quantize: Whether to apply full integer quantization
            representative_dataset: Calibration data for quantization
        
        Returns:
            Dictionary with export metadata
        """
        if not _has_tensorflow:
            raise ImportError(
                "TFLite export requires TensorFlow. "
                "Install with: pip install tensorflow"
            )
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # First export to ONNX
        onnx_path = output_path.with_suffix('.onnx')
        onnx_info = self.export_onnx(onnx_path, sample_input)
        
        # Convert ONNX to TFLite via TensorFlow
        # Note: This requires onnx-tf package
        try:
            import onnx_tf.backend as tf_backend  # type: ignore
            
            onnx_model = onnx.load(str(onnx_path))
            tf_rep = tf_backend.prepare(onnx_model)
            
            # Convert to TFLite
            converter = tf.lite.TFLiteConverter.from_concrete_functions(
                [tf_rep.signatures[tf_rep.signatures.default_key]]
            )
            
            if quantize:
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                if representative_dataset is not None:
                    def representative_gen():
                        for data in representative_dataset:
                            yield [data]
                    converter.representative_dataset = representative_gen
                    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                    converter.inference_input_type = tf.int8
                    converter.inference_output_type = tf.int8
            
            tflite_model = converter.convert()
            
            with open(output_path, 'wb') as f:
                f.write(tflite_model)
            
        except ImportError:
            # Fallback: just keep ONNX
            print("Warning: onnx-tf not available. Keeping ONNX format.")
            output_path = onnx_path
        
        model_size_mb = output_path.stat().st_size / (1024 * 1024)
        
        return {
            'path': str(output_path),
            'size_mb': model_size_mb,
            'quantized': quantize,
        }
    
    def benchmark_inference(
        self,
        sample_input: torch.Tensor | None = None,
        num_runs: int = 100,
        warmup: int = 10,
    ) -> dict[str, float]:
        """
        Benchmark model inference performance.
        
        Args:
            sample_input: Input for benchmarking
            num_runs: Number of inference runs
            warmup: Number of warmup runs
        
        Returns:
            Dictionary with timing statistics
        """
        if sample_input is None:
            sample_input = torch.randn(self.input_shape)
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = self.model(sample_input)
        
        # Benchmark
        latencies = []
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.perf_counter()
                _ = self.model(sample_input)
                end = time.perf_counter()
                latencies.append((end - start) * 1000)  # ms
        
        latencies_arr = np.array(latencies)
        
        return {
            'mean_ms': float(latencies_arr.mean()),
            'std_ms': float(latencies_arr.std()),
            'min_ms': float(latencies_arr.min()),
            'max_ms': float(latencies_arr.max()),
            'p50_ms': float(np.percentile(latencies_arr, 50)),
            'p95_ms': float(np.percentile(latencies_arr, 95)),
            'p99_ms': float(np.percentile(latencies_arr, 99)),
            'throughput_fps': float(1000.0 / latencies_arr.mean()),
        }


def export_model_for_deployment(
    model: nn.Module,
    output_dir: str | Path,
    sample_input: torch.Tensor,
    formats: list[str] | None = None,
    quantize: bool = True,
) -> dict[str, Any]:
    """
    Export model to multiple deployment formats.
    
    Args:
        model: PyTorch model
        output_dir: Directory for exports
        sample_input: Sample input tensor
        formats: List of formats ('onnx', 'torchscript', 'tflite')
        quantize: Whether to create quantized versions
    
    Returns:
        Dictionary with all export information
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if formats is None:
        formats = ['onnx', 'torchscript']
    
    exporter = ModelExporter(model, input_shape=tuple(sample_input.shape))
    
    results: dict[str, Any] = {}
    
    # Export each format
    if 'onnx' in formats:
        onnx_path = output_dir / 'model.onnx'
        results['onnx'] = exporter.export_onnx(onnx_path, sample_input)
    
    if 'torchscript' in formats:
        ts_path = output_dir / 'model.pt'
        results['torchscript'] = exporter.export_torchscript(ts_path, sample_input)
    
    if 'tflite' in formats:
        try:
            tflite_path = output_dir / 'model.tflite'
            results['tflite'] = exporter.export_tflite(tflite_path, sample_input)
        except ImportError as e:
            results['tflite'] = {'error': str(e)}
    
    # Benchmark
    results['benchmark'] = exporter.benchmark_inference(sample_input)
    
    # Model info
    results['model_info'] = {
        'num_parameters': sum(p.numel() for p in model.parameters()),
        'num_trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'input_shape': list(sample_input.shape),
    }
    
    # Save metadata
    metadata_path = output_dir / 'export_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


class ONNXModelWrapper:
    """
    Wrapper for running ONNX models with the same interface as PyTorch.
    
    Example:
        >>> wrapper = ONNXModelWrapper('model.onnx')
        >>> output = wrapper(input_tensor)  # Same as PyTorch
    """
    
    def __init__(self, onnx_path: str | Path, device: str = 'cpu'):
        """
        Initialize ONNX model wrapper.
        
        Args:
            onnx_path: Path to ONNX model
            device: Device for inference ('cpu' or 'cuda')
        """
        if not _has_onnx:
            raise ImportError("ONNX Runtime is required")
        
        self.onnx_path = Path(onnx_path)
        self.device = device
        
        # Create inference session
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(str(onnx_path), providers=providers)
        
        # Get input/output info
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_names = [o.name for o in self.session.get_outputs()]
    
    def __call__(self, x: torch.Tensor | NDArray[np.float32]) -> dict[str, NDArray[np.float32]]:
        """Run inference."""
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        
        outputs = self.session.run(self.output_names, {self.input_name: x})
        
        result = {name: output for name, output in zip(self.output_names, outputs)}
        return result
    
    def get_model_info(self) -> dict[str, Any]:
        """Get model information."""
        return {
            'input_name': self.input_name,
            'input_shape': self.input_shape,
            'output_names': self.output_names,
            'providers': self.session.get_providers(),
        }
