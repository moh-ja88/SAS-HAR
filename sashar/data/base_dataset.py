"""
Base Dataset Class for HAR Research

Provides unified interface for all HAR datasets following publication standards.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import torch
from torch.utils.data import Dataset
import numpy as np


class BaseHARDataset(ABC, Dataset):
    """
    Abstract base class for all HAR datasets.
    
    All datasets must implement this interface to ensure consistent
    evaluation protocols across experiments.
    
    Attributes:
        NUM_CLASSES: Number of activity classes
        SAMPLING_RATE: Sensor sampling rate in Hz
        SENSOR_CHANNELS: Dictionary of sensor types and channel counts
        ACTIVITY_LABELS: Mapping from class ID to activity name
    """
    
    # Class-level constants (override in subclasses)
    NUM_CLASSES: int
    SAMPLING_RATE: int
    SENSOR_CHANNELS: Dict[str, int]
    ACTIVITY_LABELS: Dict[int, str]
    
    def __init__(
        self,
        root: str,
        window_size: int = 128,
        stride: Optional[int] = None,
        subjects: Optional[List[int]] = None,
        include_boundaries: bool = True,
        transform: Optional[Any] = None,
        download: bool = False
    ):
        """
        Initialize dataset.
        
        Args:
            root: Root directory for data storage
            window_size: Number of samples per window
            stride: Stride between windows (default: window_size // 2)
            subjects: List of subject IDs to include (None = all)
            include_boundaries: Whether to include boundary labels
            transform: Optional transforms to apply
            download: Whether to download data if not present
        """
        self.root = Path(root)
        self.window_size = window_size
        self.stride = stride if stride is not None else window_size // 2
        self.subjects = subjects
        self.include_boundaries = include_boundaries
        self.transform = transform
        
        # Download if needed
        if download and not self._check_data_exists():
            self.download_and_preprocess()
        
        # Load data
        self.data, self.labels, self.boundaries, self.subject_ids = self._load_data()
        
        # Create windows
        self.windows, self.window_labels, self.window_boundaries = self._create_windows()
    
    @abstractmethod
    def _load_data(self) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray]:
        """
        Load raw data from disk.
        
        Returns:
            data: Raw sensor data [N, C, T]
            labels: Activity labels [N]
            boundaries: Boundary positions [N, T] or None
            subject_ids: Subject identifiers [N]
        """
        pass
    
    @abstractmethod
    def _check_data_exists(self) -> bool:
        """Check if processed data exists on disk."""
        pass
    
    @classmethod
    @abstractmethod
    def download_and_preprocess(cls, root: str) -> None:
        """
        Download and preprocess raw data.
        
        This method should:
        1. Download raw data from official sources
        2. Apply standard preprocessing (filtering, normalization)
        3. Save in efficient format (HDF5, NPZ, etc.)
        """
        pass
    
    def _create_windows(self) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Create sliding windows from continuous data.
        
        Returns:
            windows: Windowed data [M, C, window_size]
            labels: Window labels [M]
            boundaries: Window boundaries [M, window_size] or None
        """
        windows = []
        labels = []
        boundaries = [] if self.include_boundaries else None
        
        N, C, T = self.data.shape
        
        for i in range(0, T - self.window_size + 1, self.stride):
            window = self.data[:, :, i:i + self.window_size]
            
            # Get majority label in window
            window_label = self._get_window_label(self.labels, i, i + self.window_size)
            
            windows.append(window)
            labels.append(window_label)
            
            if self.include_boundaries and self.boundaries is not None:
                window_boundary = self.boundaries[:, i:i + self.window_size]
                boundaries.append(window_boundary)
        
        windows = np.stack(windows, axis=0)  # [M, C, window_size]
        labels = np.array(labels)
        
        if boundaries is not None:
            boundaries = np.stack(boundaries, axis=0)
        
        return windows, labels, boundaries
    
    def _get_window_label(self, labels: np.ndarray, start: int, end: int) -> int:
        """Get majority label for a window (standard protocol)."""
        window_labels = labels[start:end]
        return int(np.bincount(window_labels).argmax())
    
    def __len__(self) -> int:
        return len(self.windows)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            Dictionary containing:
                - 'data': Sensor data tensor [C, T]
                - 'label': Activity label
                - 'boundary': Boundary labels [T] (if available)
                - 'subject_id': Subject identifier
        """
        data = torch.from_numpy(self.windows[idx].astype(np.float32))
        label = torch.tensor(self.window_labels[idx], dtype=torch.long)
        
        output = {
            'data': data,
            'label': label,
            'subject_id': torch.tensor(self.subject_ids[idx], dtype=torch.long)
        }
        
        if self.include_boundaries and self.window_boundaries is not None:
            output['boundary'] = torch.from_numpy(
                self.window_boundaries[idx].astype(np.float32)
            )
        
        if self.transform:
            output = self.transform(output)
        
        return output
    
    def get_subject_split(
        self, 
        train_subjects: List[int],
        val_subjects: Optional[List[int]] = None,
        test_subjects: Optional[List[int]] = None
    ) -> Tuple[Optional['BaseHARDataset'], Optional['BaseHARDataset'], Optional['BaseHARDataset']]:
        """
        Create subject-based data splits for cross-subject evaluation.
        
        Args:
            train_subjects: Subjects for training
            val_subjects: Subjects for validation (optional)
            test_subjects: Subjects for testing (optional)
        
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        train_ds = self._filter_by_subjects(train_subjects) if train_subjects else None
        val_ds = self._filter_by_subjects(val_subjects) if val_subjects else None
        test_ds = self._filter_by_subjects(test_subjects) if test_subjects else None
        
        return train_ds, val_ds, test_ds
    
    def _filter_by_subjects(self, subjects: List[int]) -> 'BaseHARDataset':
        """Create a new dataset filtered by subject IDs."""
        # Create a shallow copy with filtered data
        filtered = self.__class__.__new__(cls=self.__class__)
        mask = np.isin(self.subject_ids, subjects)
        
        filtered.windows = self.windows[mask]
        filtered.window_labels = self.window_labels[mask]
        filtered.subject_ids = self.subject_ids[mask]
        if self.window_boundaries is not None:
            filtered.window_boundaries = self.window_boundaries[mask]
        
        return filtered
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Compute class weights for imbalanced datasets.
        
        Returns:
            Tensor of class weights [NUM_CLASSES]
        """
        class_counts = np.bincount(self.window_labels, minlength=self.NUM_CLASSES)
        class_weights = 1.0 / (class_counts + 1e-6)
        class_weights = class_weights / class_weights.sum() * self.NUM_CLASSES
        return torch.from_numpy(class_weights.astype(np.float32))
    
    def get_activity_distribution(self) -> Dict[str, float]:
        """Get distribution of activities in dataset."""
        counts = np.bincount(self.window_labels, minlength=self.NUM_CLASSES)
        total = counts.sum()
        return {
            self.ACTIVITY_LABELS.get(i, f"class_{i}"): count / total
            for i, count in enumerate(counts)
            if count > 0
        }
    
    @property
    def num_channels(self) -> int:
        """Total number of sensor channels."""
        return sum(self.SENSOR_CHANNELS.values())
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get dataset statistics for reproducibility documentation.
        
        Returns:
            Dictionary with dataset statistics
        """
        return {
            'name': self.__class__.__name__,
            'num_samples': len(self),
            'num_classes': self.NUM_CLASSES,
            'num_channels': self.num_channels,
            'window_size': self.window_size,
            'stride': self.stride,
            'sampling_rate': self.SAMPLING_RATE,
            'activity_distribution': self.get_activity_distribution(),
            'data_mean': self.windows.mean().item(),
            'data_std': self.windows.std().item(),
        }
