"""
Test suite for evaluation metrics.

Tests cover:
- Classification metrics
- Segmentation/boundary metrics
- Edge deployment metrics
"""

import pytest
import torch
import numpy as np

from sashar.evaluation.metrics import (
    ClassificationMetrics,
    SegmentationMetrics,
    EdgeMetrics,
    compute_classification_metrics,
    compute_boundary_metrics,
    compute_segment_iou,
    compute_edit_distance,
    compute_segmentation_metrics,
    compute_edge_metrics
)


class TestClassificationMetrics:
    """Test suite for classification metrics."""
    
    @pytest.fixture
    def sample_predictions(self):
        """Create sample predictions and targets."""
        # Perfect predictions
        y_true = torch.tensor([0, 1, 2, 3, 4, 5, 0, 1, 2, 3])
        y_pred = torch.tensor([0, 1, 2, 3, 4, 5, 0, 1, 2, 3])
        return y_true, y_pred
    
    @pytest.fixture
    def imperfect_predictions(self):
        """Create imperfect predictions."""
        y_true = torch.tensor([0, 1, 2, 3, 4, 5, 0, 1, 2, 3])
        y_pred = torch.tensor([0, 1, 2, 3, 3, 5, 0, 1, 1, 3])  # 2 mistakes
        return y_true, y_pred
    
    def test_perfect_predictions(self, sample_predictions):
        """Test metrics with perfect predictions."""
        y_true, y_pred = sample_predictions
        metrics = compute_classification_metrics(y_true, y_pred, num_classes=6)
        
        assert metrics.accuracy == 1.0
        assert metrics.macro_f1 == 1.0
        assert metrics.weighted_f1 == 1.0
    
    def test_imperfect_predictions(self, imperfect_predictions):
        """Test metrics with imperfect predictions."""
        y_true, y_pred = imperfect_predictions
        metrics = compute_classification_metrics(y_true, y_pred, num_classes=6)
        
        assert metrics.accuracy < 1.0
        assert metrics.accuracy > 0.5  # Should be 0.8
    
    def test_probability_predictions(self):
        """Test with probability predictions (logits)."""
        y_true = torch.tensor([0, 1, 2])
        y_pred = torch.tensor([
            [0.9, 0.05, 0.05],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8]
        ])
        
        metrics = compute_classification_metrics(y_true, y_pred, num_classes=3)
        
        assert metrics.accuracy == 1.0
    
    def test_per_class_f1(self):
        """Test per-class F1 computation."""
        y_true = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2])
        y_pred = torch.tensor([0, 0, 1, 1, 1, 2, 2, 2, 0])
        
        metrics = compute_classification_metrics(y_true, y_pred, num_classes=3)
        
        assert len(metrics.per_class_f1) == 3
        assert all(0 <= f1 <= 1 for f1 in metrics.per_class_f1)
    
    def test_confusion_matrix_shape(self, sample_predictions):
        """Test confusion matrix has correct shape."""
        y_true, y_pred = sample_predictions
        metrics = compute_classification_metrics(y_true, y_pred, num_classes=6)
        
        assert metrics.confusion_matrix.shape == (6, 6)
    
    def test_to_dict(self, sample_predictions):
        """Test conversion to dictionary."""
        y_true, y_pred = sample_predictions
        metrics = compute_classification_metrics(y_true, y_pred, num_classes=6)
        
        d = metrics.to_dict()
        
        assert 'accuracy' in d
        assert 'macro_f1' in d
        assert 'confusion_matrix' in d
    
    def test_str_representation(self, sample_predictions):
        """Test string representation."""
        y_true, y_pred = sample_predictions
        metrics = compute_classification_metrics(y_true, y_pred, num_classes=6)
        
        s = str(metrics)
        
        assert 'Accuracy' in s
        assert 'Macro F1' in s


class TestBoundaryMetrics:
    """Test suite for boundary detection metrics."""
    
    def test_perfect_boundary_detection(self):
        """Test with perfect boundary predictions."""
        pred = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0])
        true = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0])
        
        precision, recall, f1 = compute_boundary_metrics(pred, true, tolerance=0)
        
        assert precision == 1.0
        assert recall == 1.0
        assert f1 == 1.0
    
    def test_boundary_with_tolerance(self):
        """Test boundary detection with tolerance."""
        pred = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0])  # Predicted at 3
        true = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0])  # True at 4
        
        # Without tolerance, should fail
        precision, recall, f1 = compute_boundary_metrics(pred, true, tolerance=0)
        assert f1 == 0.0
        
        # With tolerance of 1, should match
        precision, recall, f1 = compute_boundary_metrics(pred, true, tolerance=1)
        assert f1 > 0
    
    def test_no_boundaries(self):
        """Test with no boundaries."""
        pred = np.zeros(10)
        true = np.zeros(10)
        
        precision, recall, f1 = compute_boundary_metrics(pred, true)
        
        # No boundaries to detect = perfect
        assert precision == 1.0
        assert recall == 1.0
    
    def test_missing_predictions(self):
        """Test with missing predictions."""
        pred = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # No predictions
        true = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0])  # Two true boundaries
        
        precision, recall, f1 = compute_boundary_metrics(pred, true)
        
        assert precision == 0.0
        assert recall == 0.0
        assert f1 == 0.0
    
    def test_false_positives(self):
        """Test with false positives."""
        pred = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])  # Many predictions
        true = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0])  # Two true boundaries
        
        precision, recall, f1 = compute_boundary_metrics(pred, true, tolerance=0)
        
        assert precision < 1.0
        assert recall == 1.0  # Found all true boundaries


class TestSegmentIoU:
    """Test suite for segment IoU computation."""
    
    def test_identical_segments(self):
        """Test IoU with identical segments."""
        pred = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        true = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        
        iou = compute_segment_iou(pred, true)
        
        assert iou == 1.0
    
    def test_shifted_boundaries(self):
        """Test IoU with shifted boundaries."""
        pred = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1])
        true = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1])
        
        iou = compute_segment_iou(pred, true)
        
        assert 0 < iou < 1  # Partial overlap


class TestEditDistance:
    """Test suite for edit distance computation."""
    
    def test_identical_sequences(self):
        """Test edit distance with identical sequences."""
        seq1 = np.array([0, 1, 2, 3])
        seq2 = np.array([0, 1, 2, 3])
        
        dist = compute_edit_distance(seq1, seq2)
        
        assert dist == 0
    
    def test_single_substitution(self):
        """Test edit distance with single substitution."""
        seq1 = np.array([0, 1, 2, 3])
        seq2 = np.array([0, 1, 5, 3])  # 2 -> 5
        
        dist = compute_edit_distance(seq1, seq2)
        
        assert dist == 1
    
    def test_single_insertion(self):
        """Test edit distance with single insertion."""
        seq1 = np.array([0, 1, 2, 3])
        seq2 = np.array([0, 1, 2, 4, 3])  # Inserted 4
        
        dist = compute_edit_distance(seq1, seq2)
        
        assert dist == 1
    
    def test_single_deletion(self):
        """Test edit distance with single deletion."""
        seq1 = np.array([0, 1, 2, 3])
        seq2 = np.array([0, 1, 3])  # Deleted 2
        
        dist = compute_edit_distance(seq1, seq2)
        
        assert dist == 1


class TestSegmentationMetrics:
    """Test suite for comprehensive segmentation metrics."""
    
    def test_perfect_segmentation(self):
        """Test with perfect segmentation."""
        pred_boundaries = torch.zeros(1, 100)
        pred_boundaries[0, 30] = 1.0
        pred_boundaries[0, 60] = 1.0
        
        true_boundaries = torch.zeros(1, 100)
        true_boundaries[0, 30] = 1.0
        true_boundaries[0, 60] = 1.0
        
        metrics = compute_segmentation_metrics(
            pred_boundaries, true_boundaries, tolerance=0
        )
        
        assert metrics.boundary_f1 == 1.0
    
    def test_segmentation_with_labels(self):
        """Test segmentation with activity labels."""
        pred_boundaries = torch.zeros(1, 100)
        pred_boundaries[0, 30] = 1.0
        
        true_boundaries = torch.zeros(1, 100)
        true_boundaries[0, 30] = 1.0
        
        pred_labels = torch.cat([
            torch.zeros(30),
            torch.ones(70)
        ])
        
        true_labels = torch.cat([
            torch.zeros(30),
            torch.ones(70)
        ])
        
        metrics = compute_segmentation_metrics(
            pred_boundaries, true_boundaries,
            pred_labels=pred_labels, true_labels=true_labels
        )
        
        assert metrics.segment_iou > 0


class TestEdgeMetrics:
    """Test suite for edge deployment metrics."""
    
    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""
        return torch.nn.Sequential(
            torch.nn.Linear(10, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 5)
        )
    
    def test_parameter_count(self, simple_model):
        """Test parameter counting."""
        metrics = compute_edge_metrics(
            simple_model,
            input_shape=(10,),
            device='cpu',
            n_runs=10
        )
        
        # Linear(10,20) = 10*20 + 20 = 220
        # Linear(20,5) = 20*5 + 5 = 105
        # Total = 325
        assert metrics.parameters == 325
    
    def test_latency_measurement(self, simple_model):
        """Test latency measurement."""
        metrics = compute_edge_metrics(
            simple_model,
            input_shape=(10,),
            device='cpu',
            n_runs=10
        )
        
        assert metrics.latency_ms > 0
        assert metrics.latency_ms < 100  # Should be fast on CPU
    
    def test_model_size(self, simple_model):
        """Test model size computation."""
        metrics = compute_edge_metrics(
            simple_model,
            input_shape=(10,),
            device='cpu',
            n_runs=10
        )
        
        assert metrics.model_size_mb > 0
    
    def test_to_dict(self, simple_model):
        """Test conversion to dictionary."""
        metrics = compute_edge_metrics(
            simple_model,
            input_shape=(10,),
            device='cpu',
            n_runs=10
        )
        
        d = metrics.to_dict()
        
        assert 'parameters' in d
        assert 'latency_ms' in d
        assert 'energy_nj' in d


class TestMetricIntegration:
    """Integration tests for metrics."""
    
    def test_full_evaluation_pipeline(self):
        """Test complete evaluation pipeline."""
        # Create sample data
        batch_size = 4
        num_classes = 6
        seq_len = 100
        
        # Classification
        y_true = torch.randint(0, num_classes, (batch_size * 10,))
        y_pred = torch.randint(0, num_classes, (batch_size * 10,))
        
        class_metrics = compute_classification_metrics(y_true, y_pred, num_classes)
        
        # Segmentation
        pred_boundaries = torch.zeros(batch_size, seq_len)
        pred_boundaries[0, 25] = 0.9
        pred_boundaries[0, 50] = 0.8
        pred_boundaries[0, 75] = 0.7
        
        true_boundaries = torch.zeros(batch_size, seq_len)
        true_boundaries[0, 25] = 1.0
        true_boundaries[0, 50] = 1.0
        true_boundaries[0, 75] = 1.0
        
        seg_metrics = compute_segmentation_metrics(
            pred_boundaries, true_boundaries, tolerance=5
        )
        
        # Verify all metrics computed
        assert class_metrics.accuracy >= 0
        assert seg_metrics.boundary_f1 >= 0


class TestMetricEdgeCases:
    """Test edge cases for metrics."""
    
    def test_single_class(self):
        """Test with single class."""
        y_true = torch.zeros(10, dtype=torch.long)
        y_pred = torch.zeros(10, dtype=torch.long)
        
        metrics = compute_classification_metrics(y_true, y_pred, num_classes=1)
        
        assert metrics.accuracy == 1.0
    
    def test_empty_boundaries(self):
        """Test with empty boundary arrays."""
        pred = np.array([])
        true = np.array([])
        
        precision, recall, f1 = compute_boundary_metrics(pred, true)
        
        # Should handle gracefully
        assert precision == 1.0  # No false positives
        assert recall == 1.0     # No missed boundaries
    
    def test_all_boundaries(self):
        """Test when everything is a boundary."""
        pred = np.ones(10)
        true = np.ones(10)
        
        precision, recall, f1 = compute_boundary_metrics(pred, true)
        
        assert precision == 1.0
        assert recall == 1.0
