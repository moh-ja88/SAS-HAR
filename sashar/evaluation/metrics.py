"""
Evaluation metrics for HAR and temporal segmentation.

Implements standard metrics for:
- Classification: Accuracy, F1, Precision, Recall, Confusion Matrix
- Segmentation: Boundary F1, Segment IoU, Edit Distance
- Edge: Latency, Energy, Memory, Parameters
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)


# ============================================================================
# Classification Metrics
# ============================================================================

@dataclass
class ClassificationMetrics:
    """Container for classification metrics."""
    accuracy: float
    macro_f1: float
    weighted_f1: float
    macro_precision: float
    macro_recall: float
    per_class_f1: np.ndarray
    confusion_matrix: np.ndarray
    
    def to_dict(self) -> Dict:
        return {
            'accuracy': self.accuracy,
            'macro_f1': self.macro_f1,
            'weighted_f1': self.weighted_f1,
            'macro_precision': self.macro_precision,
            'macro_recall': self.macro_recall,
            'per_class_f1': self.per_class_f1.tolist(),
            'confusion_matrix': self.confusion_matrix.tolist()
        }
    
    def __str__(self) -> str:
        return (
            f"ClassificationMetrics(\n"
            f"  Accuracy: {self.accuracy:.4f}\n"
            f"  Macro F1: {self.macro_f1:.4f}\n"
            f"  Weighted F1: {self.weighted_f1:.4f}\n"
            f"  Macro Precision: {self.macro_precision:.4f}\n"
            f"  Macro Recall: {self.macro_recall:.4f}\n"
            f")"
        )


def compute_classification_metrics(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    class_names: Optional[List[str]] = None,
    num_classes: Optional[int] = None
) -> ClassificationMetrics:
    """
    Compute comprehensive classification metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels (class indices or probabilities)
        class_names: Optional class names for reporting
        num_classes: Number of classes (inferred if not provided)
    
    Returns:
        ClassificationMetrics object
    """
    # Convert to numpy
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    # Handle probability predictions
    if y_pred.ndim > 1:
        y_pred = y_pred.argmax(axis=1)
    
    # Infer num_classes
    if num_classes is None:
        num_classes = max(y_true.max(), y_pred.max()) + 1
    
    # Compute metrics
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    macro_precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    
    return ClassificationMetrics(
        accuracy=accuracy,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        per_class_f1=per_class_f1,
        confusion_matrix=cm
    )


# ============================================================================
# Segmentation Metrics
# ============================================================================

@dataclass
class SegmentationMetrics:
    """Container for segmentation metrics."""
    boundary_f1: float
    boundary_precision: float
    boundary_recall: float
    segment_iou: float
    over_segmentation_rate: float
    under_segmentation_rate: float
    edit_distance: float
    mean_segment_length_error: float
    
    def to_dict(self) -> Dict:
        return {
            'boundary_f1': self.boundary_f1,
            'boundary_precision': self.boundary_precision,
            'boundary_recall': self.boundary_recall,
            'segment_iou': self.segment_iou,
            'over_segmentation_rate': self.over_segmentation_rate,
            'under_segmentation_rate': self.under_segmentation_rate,
            'edit_distance': self.edit_distance,
            'mean_segment_length_error': self.mean_segment_length_error
        }
    
    def __str__(self) -> str:
        return (
            f"SegmentationMetrics(\n"
            f"  Boundary F1: {self.boundary_f1:.4f}\n"
            f"  Boundary Precision: {self.boundary_precision:.4f}\n"
            f"  Boundary Recall: {self.boundary_recall:.4f}\n"
            f"  Segment IoU: {self.segment_iou:.4f}\n"
            f"  Over-segmentation Rate: {self.over_segmentation_rate:.4f}\n"
            f"  Under-segmentation Rate: {self.under_segmentation_rate:.4f}\n"
            f")"
        )


def compute_boundary_metrics(
    pred_boundaries: Union[np.ndarray, torch.Tensor],
    true_boundaries: Union[np.ndarray, torch.Tensor],
    tolerance: int = 10
) -> Tuple[float, float, float]:
    """
    Compute boundary detection metrics.
    
    A predicted boundary is correct if it's within `tolerance` samples
    of a true boundary.
    
    Args:
        pred_boundaries: Binary predictions (1 = boundary)
        true_boundaries: Binary ground truth (1 = boundary)
        tolerance: Acceptable error in samples
    
    Returns:
        Tuple of (precision, recall, f1)
    """
    # Convert to numpy
    if isinstance(pred_boundaries, torch.Tensor):
        pred_boundaries = pred_boundaries.cpu().numpy()
    if isinstance(true_boundaries, torch.Tensor):
        true_boundaries = true_boundaries.cpu().numpy()
    
    # Find boundary indices
    pred_indices = np.where(pred_boundaries > 0.5)[0]
    true_indices = np.where(true_boundaries > 0.5)[0]
    
    if len(pred_indices) == 0 and len(true_indices) == 0:
        return 1.0, 1.0, 1.0
    if len(pred_indices) == 0:
        return 0.0, 0.0, 0.0
    if len(true_indices) == 0:
        return 0.0, 0.0, 0.0
    
    # Compute true positives
    tp = 0
    matched_true = set()
    
    for pred_idx in pred_indices:
        # Check if any true boundary is within tolerance
        for true_idx in true_indices:
            if abs(pred_idx - true_idx) <= tolerance and true_idx not in matched_true:
                tp += 1
                matched_true.add(true_idx)
                break
    
    precision = tp / len(pred_indices)
    recall = tp / len(true_indices)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1


def compute_segment_iou(
    pred_segments: Union[np.ndarray, torch.Tensor],
    true_segments: Union[np.ndarray, torch.Tensor]
) -> float:
    """
    Compute mean segment-level IoU.
    
    Args:
        pred_segments: Predicted segment labels
        true_segments: Ground truth segment labels
    
    Returns:
        Mean IoU across all segments
    """
    if isinstance(pred_segments, torch.Tensor):
        pred_segments = pred_segments.cpu().numpy()
    if isinstance(true_segments, torch.Tensor):
        true_segments = true_segments.cpu().numpy()
    
    # Find segment boundaries
    def get_segments(labels):
        segments = []
        start = 0
        for i in range(1, len(labels)):
            if labels[i] != labels[i-1]:
                segments.append((start, i-1, labels[start]))
                start = i
        segments.append((start, len(labels)-1, labels[start]))
        return segments
    
    pred_segs = get_segments(pred_segments)
    true_segs = get_segments(true_segments)
    
    # Compute IoU for each true segment
    iou_scores = []
    
    for t_start, t_end, t_label in true_segs:
        best_iou = 0.0
        
        for p_start, p_end, p_label in pred_segs:
            if p_label != t_label:
                continue
            
            # Compute intersection and union
            inter_start = max(t_start, p_start)
            inter_end = min(t_end, p_end)
            
            if inter_start > inter_end:
                continue
            
            intersection = inter_end - inter_start + 1
            union = (t_end - t_start + 1) + (p_end - p_start + 1) - intersection
            
            iou = intersection / union if union > 0 else 0.0
            best_iou = max(best_iou, iou)
        
        iou_scores.append(best_iou)
    
    return np.mean(iou_scores) if iou_scores else 0.0


def compute_edit_distance(
    pred_sequence: Union[np.ndarray, torch.Tensor],
    true_sequence: Union[np.ndarray, torch.Tensor]
) -> int:
    """
    Compute Levenshtein edit distance between sequences.
    
    Args:
        pred_sequence: Predicted activity sequence
        true_sequence: Ground truth activity sequence
    
    Returns:
        Edit distance (number of insertions, deletions, substitutions)
    """
    if isinstance(pred_sequence, torch.Tensor):
        pred_sequence = pred_sequence.cpu().numpy()
    if isinstance(true_sequence, torch.Tensor):
        true_sequence = true_sequence.cpu().numpy()
    
    m, n = len(pred_sequence), len(true_sequence)
    
    # Create DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_sequence[i-1] == true_sequence[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],     # Deletion
                    dp[i][j-1],     # Insertion
                    dp[i-1][j-1]    # Substitution
                )
    
    return dp[m][n]


def compute_segmentation_metrics(
    pred_boundaries: Union[np.ndarray, torch.Tensor],
    true_boundaries: Union[np.ndarray, torch.Tensor],
    pred_labels: Optional[Union[np.ndarray, torch.Tensor]] = None,
    true_labels: Optional[Union[np.ndarray, torch.Tensor]] = None,
    tolerance: int = 10
) -> SegmentationMetrics:
    """
    Compute comprehensive segmentation metrics.
    
    Args:
        pred_boundaries: Predicted boundary scores (0-1)
        true_boundaries: Ground truth boundaries (binary)
        pred_labels: Optional predicted activity labels
        true_labels: Optional ground truth activity labels
        tolerance: Boundary tolerance in samples
    
    Returns:
        SegmentationMetrics object
    """
    # Convert to numpy
    if isinstance(pred_boundaries, torch.Tensor):
        pred_boundaries = pred_boundaries.cpu().numpy()
    if isinstance(true_boundaries, torch.Tensor):
        true_boundaries = true_boundaries.cpu().numpy()
    
    # Binarize predictions
    pred_binary = (pred_boundaries > 0.5).astype(int)
    
    # Boundary metrics
    b_precision, b_recall, b_f1 = compute_boundary_metrics(
        pred_binary, true_boundaries, tolerance
    )
    
    # Segment IoU (if labels provided)
    if pred_labels is not None and true_labels is not None:
        segment_iou = compute_segment_iou(pred_labels, true_labels)
        edit_dist = compute_edit_distance(pred_labels, true_labels)
    else:
        segment_iou = 0.0
        edit_dist = 0
    
    # Over/under-segmentation rates
    pred_count = pred_binary.sum()
    true_count = true_boundaries.sum()
    
    if true_count > 0:
        over_seg = max(0, (pred_count - true_count) / true_count)
        under_seg = max(0, (true_count - pred_count) / true_count)
    else:
        over_seg = 0.0
        under_seg = 0.0
    
    # Mean segment length error (placeholder)
    mean_len_error = 0.0
    
    return SegmentationMetrics(
        boundary_f1=b_f1,
        boundary_precision=b_precision,
        boundary_recall=b_recall,
        segment_iou=segment_iou,
        over_segmentation_rate=over_seg,
        under_segmentation_rate=under_seg,
        edit_distance=edit_dist,
        mean_segment_length_error=mean_len_error
    )


# ============================================================================
# Edge Metrics
# ============================================================================

@dataclass
class EdgeMetrics:
    """Container for edge deployment metrics."""
    parameters: int
    parameters_m: float  # In millions
    flops: int
    flops_m: float  # In millions
    latency_ms: float
    energy_nj: float
    peak_memory_kb: float
    model_size_mb: float
    
    def to_dict(self) -> Dict:
        return {
            'parameters': self.parameters,
            'parameters_m': self.parameters_m,
            'flops': self.flops,
            'flops_m': self.flops_m,
            'latency_ms': self.latency_ms,
            'energy_nj': self.energy_nj,
            'peak_memory_kb': self.peak_memory_kb,
            'model_size_mb': self.model_size_mb
        }
    
    def __str__(self) -> str:
        return (
            f"EdgeMetrics(\n"
            f"  Parameters: {self.parameters_m:.2f}M\n"
            f"  FLOPs: {self.flops_m:.2f}M\n"
            f"  Latency: {self.latency_ms:.2f}ms\n"
            f"  Energy: {self.energy_nj:.2f}nJ\n"
            f"  Memory: {self.peak_memory_kb:.1f}KB\n"
            f"  Size: {self.model_size_mb:.2f}MB\n"
            f")"
        )


def compute_edge_metrics(
    model: torch.nn.Module,
    input_shape: Tuple[int, ...],
    device: str = 'cpu',
    n_runs: int = 100
) -> EdgeMetrics:
    """
    Compute edge deployment metrics.
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape (without batch dimension)
        device: Device for latency measurement
        n_runs: Number of runs for latency measurement
    
    Returns:
        EdgeMetrics object
    """
    model = model.to(device)
    model.eval()
    
    # Count parameters
    params = sum(p.numel() for p in model.parameters())
    params_m = params / 1e6
    
    # Model size
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    model_size_mb = (param_size + buffer_size) / (1024 ** 2)
    
    # Estimate FLOPs (simplified)
    # For CNN + Transformer: approximate based on architecture
    dummy_input = torch.randn(1, *input_shape).to(device)
    
    # Use profiling to estimate FLOPs
    flops = _estimate_flops(model, dummy_input)
    flops_m = flops / 1e6
    
    # Measure latency
    with torch.no_grad():
        # Warmup
        for _ in range(10):
            _ = model(dummy_input)
        
        # Measure
        import time
        latencies = []
        for _ in range(n_runs):
            start = time.perf_counter()
            _ = model(dummy_input)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to ms
        
        latency_ms = np.mean(latencies)
    
    # Estimate energy (using 45nJ/op as reference for ARM Cortex-M4)
    # This is an approximation - real measurement requires hardware
    energy_nj = flops * 45  # nJ per operation
    
    # Peak memory (approximate from activations)
    peak_memory_kb = _estimate_peak_memory(model, dummy_input)
    
    return EdgeMetrics(
        parameters=params,
        parameters_m=params_m,
        flops=flops,
        flops_m=flops_m,
        latency_ms=latency_ms,
        energy_nj=energy_nj,
        peak_memory_kb=peak_memory_kb,
        model_size_mb=model_size_mb
    )


def _estimate_flops(model: torch.nn.Module, input_tensor: torch.Tensor) -> int:
    """Estimate FLOPs for a model."""
    total_flops = 0
    
    def hook_fn(module, input, output):
        nonlocal total_flops
        
        if isinstance(module, torch.nn.Conv1d):
            # Conv1d: 2 * Cin * Cout * K * L_out
            batch_size = input[0].size(0)
            cout = module.out_channels
            cin = module.in_channels // module.groups
            k = module.kernel_size[0]
            l_out = output.size(2)
            total_flops += 2 * batch_size * cin * cout * k * l_out
            
        elif isinstance(module, torch.nn.Linear):
            # Linear: 2 * batch * in * out
            batch_size = input[0].size(0)
            total_flops += 2 * batch_size * input[0].size(-1) * output.size(-1)
            
        elif isinstance(module, (torch.nn.MultiheadAttention,)):
            # Attention: ~4 * L * D^2 for QKV + output projection
            # This is simplified
            batch_size = input[0].size(1)
            seq_len = input[0].size(0)
            embed_dim = module.embed_dim
            total_flops += 4 * batch_size * seq_len * embed_dim * embed_dim
    
    hooks = []
    for module in model.modules():
        if isinstance(module, (torch.nn.Conv1d, torch.nn.Linear, torch.nn.MultiheadAttention)):
            hooks.append(module.register_forward_hook(hook_fn))
    
    with torch.no_grad():
        _ = model(input_tensor)
    
    for hook in hooks:
        hook.remove()
    
    return total_flops


def _estimate_peak_memory(model: torch.nn.Module, input_tensor: torch.Tensor) -> float:
    """Estimate peak memory usage in KB."""
    # Simplified estimation based on activation sizes
    total_activation_size = 0
    
    def hook_fn(module, input, output):
        nonlocal total_activation_size
        total_activation_size += output.numel() * output.element_size()
    
    hooks = []
    for module in model.modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            hooks.append(module.register_forward_hook(hook_fn))
    
    with torch.no_grad():
        _ = model(input_tensor)
    
    for hook in hooks:
        hook.remove()
    
    return total_activation_size / 1024  # Convert to KB


# ============================================================================
# Utility Functions
# ============================================================================

def print_metrics_summary(
    class_metrics: Optional[ClassificationMetrics] = None,
    seg_metrics: Optional[SegmentationMetrics] = None,
    edge_metrics: Optional[EdgeMetrics] = None
):
    """Print a formatted summary of all metrics."""
    print("=" * 60)
    print("METRICS SUMMARY")
    print("=" * 60)
    
    if class_metrics:
        print("\nClassification:")
        print(f"  Accuracy:  {class_metrics.accuracy:.4f}")
        print(f"  Macro F1:  {class_metrics.macro_f1:.4f}")
        print(f"  Weighted F1: {class_metrics.weighted_f1:.4f}")
    
    if seg_metrics:
        print("\nSegmentation:")
        print(f"  Boundary F1: {seg_metrics.boundary_f1:.4f}")
        print(f"  Segment IoU: {seg_metrics.segment_iou:.4f}")
        print(f"  Edit Distance: {seg_metrics.edit_distance}")
    
    if edge_metrics:
        print("\nEdge Deployment:")
        print(f"  Parameters: {edge_metrics.parameters_m:.2f}M")
        print(f"  Latency: {edge_metrics.latency_ms:.2f}ms")
        print(f"  Energy: {edge_metrics.energy_nj:.2f}nJ")
    
    print("=" * 60)



def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    boundaries_true: Optional[np.ndarray] = None,
    boundaries_pred: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute all available metrics in a single call.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        boundaries_true: Ground truth boundaries (optional)
        boundaries_pred: Predicted boundaries (optional)
        class_names: List of class names (optional)
    
    Returns:
        Dictionary containing all computed metrics
    """
    results = {}
    
    # Classification metrics
    class_metrics = compute_classification_metrics(y_true, y_pred, class_names)
    results['classification'] = {
        'accuracy': class_metrics.accuracy,
        'macro_f1': class_metrics.macro_f1,
        'weighted_f1': class_metrics.weighted_f1,
        'macro_precision': class_metrics.macro_precision,
        'macro_recall': class_metrics.macro_recall,
        'per_class_f1': class_metrics.per_class_f1,
    }
    
    # Boundary metrics if provided
    if boundaries_true is not None and boundaries_pred is not None:
        bnd_metrics = compute_boundary_metrics(boundaries_true, boundaries_pred)
        results['boundary'] = {
            'precision': bnd_metrics.precision,
            'recall': bnd_metrics.recall,
            'f1': bnd_metrics.f1,
        }
    
    return results

