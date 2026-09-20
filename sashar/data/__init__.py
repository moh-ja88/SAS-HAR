"""
SAS-HAR Data Module

Provides unified dataset loaders for HAR research.
"""

from sashar.data.base_dataset import BaseHARDataset
from sashar.data.uci_har import UCIHardataset
from sashar.data.wisdm import WISDMDataset
from sashar.data.pamap2 import PAMAP2Dataset
from sashar.data.opportunity import OpportunityDataset

# Try to import transforms if available
try:
    from sashar.data.transforms import (
        Compose,
        Normalize,
        Jitter,
        Scaling,
        TimeWarp,
        RandomRotation,
        ChannelDropout
    )
    _HAS_TRANSFORMS = True
except ImportError:
    _HAS_TRANSFORMS = False

__all__ = [
    'BaseHARDataset',
    'UCIHardataset',
    'WISDMDataset',
    'PAMAP2Dataset',
    'OpportunityDataset',
    'get_dataset',
    'DATASET_REGISTRY'
]

if _HAS_TRANSFORMS:
    __all__.extend([
        'Compose',
        'Normalize',
        'Jitter',
        'Scaling',
        'TimeWarp',
        'RandomRotation',
        'ChannelDropout'
    ])


DATASET_REGISTRY = {
    'uci_har': UCIHardataset,
    'wisdm': WISDMDataset,
    'pamap2': PAMAP2Dataset,
    'opportunity': OpportunityDataset
}


def get_dataset(name: str, **kwargs):
    """
    Get a dataset by name.
    
    Args:
        name: Dataset name ('uci_har', 'wisdm', 'pamap2', 'opportunity')
        **kwargs: Arguments to pass to dataset constructor
    
    Returns:
        Dataset instance
    
    Example:
        >>> dataset = get_dataset('uci_har', root='data/', split='train')
    """
    if name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset: {name}. "
            f"Available: {list(DATASET_REGISTRY.keys())}"
        )
    
    return DATASET_REGISTRY[name](**kwargs)
