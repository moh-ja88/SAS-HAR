"""
RealWorld HAR Dataset Loader

A comprehensive dataset loader for the RealWorld (RWHAR) dataset, supporting
multiple body positions and sensor modalities.

Dataset Details:
- 15 subjects
- 8 activities
- 7 body positions (chest, forearm, head, shin, thigh, upper arm, waist)
- Sensors: accelerometer, gyroscope, magnetometer
- Sampling rate: 50 Hz

URL: https://sensor.informatik.uni-mannheim.de/#dataset_realworld
"""

import os
import zipfile
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch

from ..base_dataset import BaseHARDataset


class RealWorldHARDataset(BaseHARDataset):
    """
    RealWorld HAR Dataset Loader
    
    The RealWorld (RWHAR) dataset is a publicly available dataset for 
    human activity recognition using wearable sensors. It contains data
    from 15 subjects performing 8 activities with sensors placed at 
    7 different body positions.
    
    Attributes:
        NUM_CLASSES: 8 activity classes
        SAMPLING_RATE: 50 Hz
        SENSOR_CHANNELS: Dictionary of sensor types and channel counts
        ACTIVITY_LABELS: Mapping from class ID to activity name
        POSITIONS: List of valid body positions
    
    Example:
        >>> dataset = RealWorldHARDataset(
        ...     root='data/realworld',
        ...     position='waist',
        ...     download=True
        ... )
        >>> print(len(dataset))
        >>> sample = dataset[0]
    """
    
    # Class-level constants
    NUM_CLASSES: int = 8
    SAMPLING_RATE: int = 50
    
    SENSOR_CHANNELS: Dict[str, int] = {
        'accelerometer': 3,
        'gyroscope': 3,
        'magnetometer': 3
    }
    
    ACTIVITY_LABELS: Dict[int, str] = {
        0: 'walking',
        1: 'running',
        2: 'sitting',
        3: 'standing',
        4: 'lying',
        5: 'climbing_up',
        6: 'climbing_down',
        7: 'jumping'
    }
    
    POSITIONS: List[str] = [
        'chest',
        'forearm',
        'head',
        'shin',
        'thigh',
        'upper_arm',
        'waist'
    ]
    
    SUBJECTS: List[int] = list(range(1, 16))  # 15 subjects
    
    # Dataset URLs
    DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00505/"
    PROXIED_URL = "https://sensor.informatik.uni-mannheim.de/dataset/RealWorldHAR/"
    
    def __init__(
        self,
        root: str,
        position: str = 'waist',
        sensors: Optional[List[str]] = None,
        window_size: int = 128,
        stride: Optional[int] = None,
        subjects: Optional[List[int]] = None,
        include_boundaries: bool = True,
        transform: Optional[object] = None,
        download: bool = False,
        use_cached: bool = True
    ):
        """
        Initialize RealWorld HAR dataset.
        
        Args:
            root: Root directory for data storage
            position: Body position ('chest', 'forearm', 'head', 'shin', 
                     'thigh', 'upper_arm', 'waist')
            sensors: List of sensors to use ('accelerometer', 'gyroscope', 
                    'magnetometer'). None = all sensors
            window_size: Number of samples per window
            stride: Stride between windows (default: window_size // 2)
            subjects: List of subject IDs to include (None = all 15)
            include_boundaries: Whether to include boundary labels
            transform: Optional transforms to apply
            download: Whether to download data if not present
            use_cached: Whether to use cached preprocessed data
        """
        self.position = position.lower()
        self.sensors = sensors or list(self.SENSOR_CHANNELS.keys())
        self.use_cached = use_cached
        
        # Validate position
        if self.position not in self.POSITIONS:
            raise ValueError(
                f"Invalid position '{position}'. "
                f"Must be one of: {self.POSITIONS}"
            )
        
        # Validate sensors
        for sensor in self.sensors:
            if sensor not in self.SENSOR_CHANNELS:
                raise ValueError(
                    f"Invalid sensor '{sensor}'. "
                    f"Must be one of: {list(self.SENSOR_CHANNELS.keys())}"
                )
        
        # Filter subjects if specified
        if subjects is not None:
            self.subjects = [s for s in subjects if s in self.SUBJECTS]
            if not self.subjects:
                raise ValueError("No valid subjects specified")
        else:
            self.subjects = self.SUBJECTS
        
        super().__init__(
            root=root,
            window_size=window_size,
            stride=stride,
            subjects=subjects,
            include_boundaries=include_boundaries,
            transform=transform,
            download=download
        )
    
    @property
    def num_channels(self) -> int:
        """Total number of sensor channels."""
        return sum(self.SENSOR_CHANNELS[s] for s in self.sensors)
    
    def _check_data_exists(self) -> bool:
        """Check if processed data exists on disk."""
        processed_path = self.root / 'processed' / f'{self.position}_data.npz'
        return processed_path.exists()
    
    def _load_data(self) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray]:
        """
        Load RealWorld HAR data from disk.
        
        Returns:
            data: Raw sensor data [N, C, T]
            labels: Activity labels [N]
            boundaries: Boundary positions [N, T] or None
            subject_ids: Subject identifiers [N]
        """
        processed_path = self.root / 'processed'
        data_file = processed_path / f'{self.position}_data.npz'
        
        if not data_file.exists():
            raise FileNotFoundError(
                f"Processed data not found at {data_file}. "
                "Call download_and_preprocess() first."
            )
        
        # Load preprocessed data
        loaded = np.load(data_file, allow_pickle=True)
        
        data = loaded['data']
        labels = loaded['labels']
        subject_ids = loaded['subject_ids']
        boundaries = loaded.get('boundaries', None)
        
        return data, labels, boundaries, subject_ids
    
    @classmethod
    def download_and_preprocess(cls, root: str, position: str = 'waist') -> None:
        """
        Download and preprocess RealWorld HAR dataset.
        
        This method:
        1. Downloads raw data from official sources
        2. Applies standard preprocessing (filtering, normalization)
        3. Saves in efficient NPZ format
        
        Args:
            root: Root directory for data storage
            position: Body position to process
        """
        root = Path(root)
        raw_dir = root / 'raw'
        processed_dir = root / 'processed'
        
        # Create directories
        raw_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Downloading RealWorld HAR dataset to {raw_dir}...")
        
        # Download data (implementation depends on actual data source)
        cls._download_raw_data(raw_dir)
        
        print("Preprocessing data...")
        
        # Process each subject and activity
        all_data = []
        all_labels = []
        all_subjects = []
        all_boundaries = []
        
        for subject_id in cls.SUBJECTS:
            subject_data = cls._load_subject_data(raw_dir, subject_id, position)
            
            for activity_id, activity_name in cls.ACTIVITY_LABELS.items():
                if activity_name in subject_data:
                    sensor_data = subject_data[activity_name]
                    
                    # Preprocess
                    processed = cls._preprocess_sensor_data(sensor_data)
                    
                    # Create windows
                    windows, boundaries = cls._create_windows_from_continuous(
                        processed, 
                        activity_id
                    )
                    
                    all_data.append(windows)
                    all_labels.extend([activity_id] * len(windows))
                    all_subjects.extend([subject_id] * len(windows))
                    all_boundaries.extend(boundaries)
        
        # Concatenate all data
        data = np.concatenate(all_data, axis=0)
        labels = np.array(all_labels)
        subject_ids = np.array(all_subjects)
        boundaries = np.array(all_boundaries) if all_boundaries else None
        
        # Save processed data
        output_file = processed_dir / f'{position}_data.npz'
        np.savez(
            output_file,
            data=data,
            labels=labels,
            subject_ids=subject_ids,
            boundaries=boundaries
        )
        
        print(f"Preprocessed data saved to {output_file}")
        print(f"Total samples: {len(data)}")
        print(f"Shape: {data.shape}")
    
    @classmethod
    def _download_raw_data(cls, raw_dir: Path) -> None:
        """Download raw data from official sources."""
        # Check if already downloaded
        if (raw_dir / 'download_complete').exists():
            print("Raw data already downloaded.")
            return
        
        # Implementation for actual download
        # The RealWorld dataset requires specific download process
        # This is a placeholder - actual implementation depends on 
        # data availability
        
        try:
            # Try to download from UCI ML repository
            zip_path = raw_dir / 'RealWorldHAR.zip'
            
            if not zip_path.exists():
                print(f"Downloading from {cls.PROXIED_URL}...")
                response = requests.get(cls.PROXIED_URL, stream=True)
                response.raise_for_status()
                
                with open(zip_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            
            # Extract
            print("Extracting...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(raw_dir)
            
            # Mark download complete
            (raw_dir / 'download_complete').touch()
            
        except requests.RequestException as e:
            print(f"Download failed: {e}")
            print("Please download manually from:")
            print("https://sensor.informatik.uni-mannheim.de/#dataset_realworld")
            raise
    
    @classmethod
    def _load_subject_data(
        cls, 
        raw_dir: Path, 
        subject_id: int, 
        position: str
    ) -> Dict[str, np.ndarray]:
        """
        Load data for a specific subject and position.
        
        Args:
            raw_dir: Directory containing raw data
            subject_id: Subject ID (1-15)
            position: Body position
        
        Returns:
            Dictionary mapping activity names to sensor data arrays
        """
        subject_dir = raw_dir / f'proband{subject_id}' / 'data'
        position_dir = subject_dir / position
        
        if not position_dir.exists():
            print(f"Warning: No data for subject {subject_id}, position {position}")
            return {}
        
        data = {}
        
        for activity_id, activity_name in cls.ACTIVITY_LABELS.items():
            activity_file = position_dir / f'{activity_name}.csv'
            
            if activity_file.exists():
                # Load CSV data
                raw = np.loadtxt(activity_file, delimiter=',')
                data[activity_name] = raw
        
        return data
    
    @classmethod
    def _preprocess_sensor_data(cls, data: np.ndarray) -> np.ndarray:
        """
        Apply standard preprocessing to sensor data.
        
        Steps:
        1. Resample to 50 Hz if needed
        2. Apply low-pass filter (20 Hz cutoff)
        3. Normalize to [-1, 1] range
        4. Handle missing values
        
        Args:
            data: Raw sensor data [T, C]
        
        Returns:
            Preprocessed data [T, C]
        """
        from scipy import signal
        
        # Handle NaN values
        data = np.nan_to_num(data, nan=0.0)
        
        # Low-pass filter (20 Hz cutoff at 50 Hz sampling)
        if data.shape[0] > 10:  # Need enough samples for filtering
            b, a = signal.butter(4, 20 / (cls.SAMPLING_RATE / 2), btype='low')
            data = signal.filtfilt(b, a, data, axis=0)
        
        # Normalize to [-1, 1]
        data_max = np.abs(data).max(axis=0, keepdims=True) + 1e-8
        data = data / data_max
        
        return data
    
    @classmethod
    def _create_windows_from_continuous(
        cls, 
        data: np.ndarray, 
        activity_id: int,
        window_size: int = 128,
        stride: int = 64
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Create windows from continuous data.
        
        Args:
            data: Continuous sensor data [T, C]
            activity_id: Activity label for all windows
            window_size: Window size in samples
            stride: Stride between windows
        
        Returns:
            windows: Windowed data [N, C, window_size]
            boundaries: Boundary labels for each window
        """
        T, C = data.shape
        windows = []
        boundaries = []
        
        for i in range(0, T - window_size + 1, stride):
            window = data[i:i + window_size]
            
            # Transpose to [C, T]
            window = window.T
            
            windows.append(window)
            
            # Create boundary labels (0 everywhere for single-activity window)
            boundary = np.zeros(window_size)
            boundaries.append(boundary)
        
        if not windows:
            return np.array([]).reshape(0, C, window_size), []
        
        return np.stack(windows), boundaries
    
    def get_subject_split(
        self,
        train_subjects: Optional[List[int]] = None,
        val_subjects: Optional[List[int]] = None,
        test_subjects: Optional[List[int]] = None
    ) -> Tuple[Optional['RealWorldHARDataset'], 
               Optional['RealWorldHARDataset'], 
               Optional['RealWorldHARDataset']]:
        """
        Create subject-based data splits for cross-subject evaluation.
        
        Args:
            train_subjects: Subjects for training (default: [1-10])
            val_subjects: Subjects for validation (default: [11-12])
            test_subjects: Subjects for testing (default: [13-15])
        
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        # Default LOSO-style split
        if train_subjects is None:
            train_subjects = list(range(1, 11))
        if val_subjects is None:
            val_subjects = [11, 12]
        if test_subjects is None:
            test_subjects = [13, 14, 15]
        
        return super().get_subject_split(train_subjects, val_subjects, test_subjects)
    
    def get_activity_distribution(self) -> Dict[str, float]:
        """Get distribution of activities in dataset."""
        counts = np.bincount(self.window_labels, minlength=self.NUM_CLASSES)
        total = counts.sum()
        return {
            self.ACTIVITY_LABELS.get(i, f"class_{i}"): count / total
            for i, count in enumerate(counts)
            if count > 0
        }
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics for reproducibility documentation."""
        stats = super().get_statistics()
        stats.update({
            'position': self.position,
            'sensors': self.sensors,
            'num_subjects': len(np.unique(self.subject_ids)),
        })
        return stats


# Convenience function for common use cases
def create_realworld_loaders(
    root: str,
    position: str = 'waist',
    batch_size: int = 64,
    **kwargs
) -> Tuple[torch.utils.data.DataLoader, 
           torch.utils.data.DataLoader, 
           torch.utils.data.DataLoader]:
    """
    Create train/val/test dataloaders for RealWorld HAR.
    
    Args:
        root: Root directory for data
        position: Body position
        batch_size: Batch size for all loaders
        **kwargs: Additional arguments passed to dataset
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    from torch.utils.data import DataLoader
    
    # Create full dataset to trigger download if needed
    full_dataset = RealWorldHARDataset(
        root=root,
        position=position,
        download=True,
        **kwargs
    )
    
    # Create splits
    train_ds, val_ds, test_ds = full_dataset.get_subject_split()
    
    # Create loaders
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=4
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=4
    )
    
    return train_loader, val_loader, test_loader


# Dataset info for documentation
DATASET_INFO = {
    'name': 'RealWorld HAR',
    'abbreviation': 'RWHAR',
    'year': 2018,
    'subjects': 15,
    'activities': 8,
    'positions': 7,
    'sensors': ['accelerometer', 'gyroscope', 'magnetometer'],
    'sampling_rate': 50,
    'duration_hours': 9.5,
    'url': 'https://sensor.informatik.uni-mannheim.de/#dataset_realworld',
    'license': 'CC BY 4.0',
    'citation': '''
@inproceedings{sztyler2016body,
  title={On-body localization of wearable devices: An investigation of position-aware activity recognition},
  author={Sztyler, Timo and Stuckenschmidt, Heiner},
  booktitle={2016 IEEE International Conference on Pervasive Computing and Communications (PerCom)},
  pages={1--9},
  year={2016},
  organization={IEEE}
}
'''
}
