"""
SAS-HAR Evaluation Module

Provides metrics and analysis tools for HAR evaluation.
"""

from sashar.evaluation.metrics import (
    # Classification
    ClassificationMetrics,
    compute_classification_metrics,
    # Segmentation
    SegmentationMetrics,
    compute_segmentation_metrics,
    compute_boundary_metrics,
    compute_segment_iou,
    compute_edit_distance,
    # Edge
    EdgeMetrics,
    compute_edge_metrics,
    # Utilities
    print_metrics_summary
)

__all__ = [
    # Classification
    'ClassificationMetrics',
    'compute_classification_metrics',
    # Segmentation
    'SegmentationMetrics',
    'compute_segmentation_metrics',
    'compute_boundary_metrics',
    'compute_segment_iou',
    'compute_edit_distance',
    # Edge
    'EdgeMetrics',
    'compute_edge_metrics',
    # Utilities
    'print_metrics_summary'
]
