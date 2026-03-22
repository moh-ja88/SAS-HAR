"""
SAS-HAR Framework
Self-Supervised Attention-based Segmentation for Human Activity Recognition

Author: Mohammed Jasim
Supervisor: Dr. Mohd Halim Mohd Noor
Institution: Universiti Sains Malaysia
Year: 2025-2029
"""

__version__ = "0.1.0"
__author__ = "Mohammed Jasim"

from sashar.models.sas_har import SASHAR
from sashar.models.encoder import CNNFeatureEncoder, TransformerTemporalModule
from sashar.models.heads import BoundaryHead, ClassificationHead, MultiTaskHead
from sashar.models.tcbl import TCBLPretrainer, ActivityAugmentation

# Aliases for backward compatibility
CNNEncoder = CNNFeatureEncoder
TransformerEncoder = TransformerTemporalModule

__all__ = [
    "SASHAR",
    "CNNEncoder",
    "TransformerEncoder",
    "CNNFeatureEncoder",
    "TransformerTemporalModule",
    "BoundaryHead",
    "ClassificationHead",
    "MultiTaskHead",
    "TCBLPretrainer",
    "ActivityAugmentation",
]
