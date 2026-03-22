"""
SAS-HAR Baselines Module

Provides baseline methods for HAR comparison.
"""

from sashar.baselines.segmentation_baselines import (
    FixedSlidingWindow,
    AdaptiveSlidingWindow,
    SimilaritySegmentation,
    DeepSimilaritySegmentation,
    BASELINE_REGISTRY,
    get_baseline
)

__all__ = [
    'FixedSlidingWindow',
    'AdaptiveSlidingWindow',
    'SimilaritySegmentation',
    'DeepSimilaritySegmentation',
    'BASELINE_REGISTRY',
    'get_baseline'
]
