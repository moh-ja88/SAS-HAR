"""
PAMAP2 (Physical Activity Monitoring in the Ageing Population) Dataset

Dataset characteristics:
- 9 subjects
- 18 activities (12 mandatory + 6 optional)
- 3 IMUs (chest, wrist, ankle) + heart rate monitor
- 100 Hz sampling rate
- Continuous recordings with protocol
"""

from pathlib import Path
from typing import Optional, List, Dict
import numpy as np
import torch
from sashar.data.base_dataset import BaseHARDataset


class PAMAP2Dataset(BaseHARDataset):
    """PAMAP2 Dataset loader for multi-sensor HAR research."""
    
    NUM_CLASSES = 12  # Protocol activities only
    SAMPLING_RATE = 100
    SENSOR_CHANNELS = {
        'accel_hand': 3, 'gyro_hand': 3, 'mag_hand': 3,
        'accel_chest': 3, 'gyro_chest': 3, 'mag_chest': 3,
        'accel_ankle': 3, 'gyro_ankle': 3, 'mag_ankle': 3,
        'heart_rate': 1
    }  # Total: 28 channels
    
    ACTIVITY_LABELS = {
        0: 'other',
        1: 'lying',
        2: 'sitting',
        3: 'standing',
        4: 'walking',
        5: 'running',
        6: 'cycling',
        7: 'nordic_walking',
        12: 'ascending_stairs',
        13: 'descending_stairs',
        16: 'vacuum_cleaning',
        17: 'ironing',
        24: 'rope_jumping'
    }
    
    # Activity groups for evaluation
    STATIC_ACTIVITIES = [1, 2, 3, 17]  # lying, sitting, standing, ironing
    DYNAMIC_ACTIVITIES = [4, 5, 6, 7, 12, 13, 16, 24]
    TRANSITIONAL_ACTIVITIES = []  # Not explicitly labeled in PAMAP2
    
    def __init__(
        self,
        root: str,
        window_size: int = 512,  # 5.12 seconds @ 100Hz
        stride: int = 256,   # 50% overlap
        split: str = 'train',
        subjects: Optional[List[int]] = None,
        use_channels: Optional[List[str]] = None,
        use_norm: bool = True,
        download: bool = False,
        include_optional: bool = False,
        use_heart_rate: bool = True
    ):
        """
        Args:
            root: Root directory containing the dataset
            window_size: Window size in samples (default: 512 = 5.12s @ 100Hz)
            stride: Stride between windows (default: 256 = 50% overlap)
            split: One of 'train', 'test', 'val'
            subjects: Optional list of subject IDs (1-9)
            use_channels: Which IMU positions to use ['hand', 'chest', 'ankle']
            use_norm: Whether to apply normalization
            download: Whether to download dataset if not found
            include_optional: Include optional protocol activities
            use_heart_rate: Include heart rate channel
        """
        super().__init__(root, window_size, stride, split, subjects, use_norm)
        
        self.include_optional = include_optional
        self.use_heart_rate = use_heart_rate
        self.use_channels = use_channels or ['hand', 'chest', 'ankle']
        
        self._load_data()
    
    def _load_data(self):
        """Load and preprocess PAMAP2 data."""
        root_path = Path(self.root)
        protocol_path = root_path / "PAMAP2_Dataset" / "Protocol"
        
        if not protocol_path.exists():
            if self.download:
                self._download_dataset(root_path)
            else:
                raise FileNotFoundError(
                    f"PAMAP2 dataset not found at {root_path}. "
                    f"Set download=True to automatically download."
                )
        
        # Load all subject data
        all_data = []
        all_labels = []
        all_subjects = []
        all_boundaries = []
        
        subject_ids = self.subjects if self.subjects else range(101, 110)
        
        for subject_id in subject_ids:
            # Load subject data
            subject_files = list(protocol_path.glob(f"subject{subject_id}.dat"))
            
            if not subject_files:
                continue
            
            data = self._load_subject_file(subject_files[0])
            
            if data is None:
                continue
            
            # Extract windows
            windows, labels, boundaries = self._extract_windows(data)
            
            all_data.append(windows)
            all_labels.append(labels)
            all_boundaries.append(boundaries)
            all_subjects.extend([subject_id - 100] * len(windows))  # 1-indexed subjects
        
        # Concatenate all data
        self.data = torch.from_numpy(np.concatenate(all_data)).float()
        self.labels = torch.from_numpy(np.concatenate(all_labels)).long()
        self.boundaries = torch.from_numpy(np.concatenate(all_boundaries)).float()
        self.subject_ids = torch.tensor(all_subjects).long()
        
        # Apply normalization
        if self.use_norm:
            self._normalize_data()
    
    def _load_subject_file(self, filepath: Path) -> Optional[np.ndarray]:
        """Load a single subject's data file."""
        try:
            # PAMAP2 has 54 columns per row
            # Column structure defined in dataset README
            data = np.loadtxt(filepath, dtype=np.float32)
            
            # Handle NaN values
            data = self._handle_missing_values(data)
            
            return data
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    
    def _handle_missing_values(self, data: np.ndarray) -> np.ndarray:
        """Handle missing values in PAMAP2 data."""
        # Heart rate column (index 2) has ~50% missing
        # Use forward fill + backward fill
        
        for col in range(data.shape[1]):
            col_data = data[:, col]
            nan_mask = np.isnan(col_data)
            
            if nan_mask.all():
                # All NaN - replace with zeros
                data[:, col] = 0.0
            else:
                # Forward fill
                last_valid = None
                for i in range(len(col_data)):
                    if not nan_mask[i]:
                        last_valid = col_data[i]
                    elif last_valid is not None:
                        data[i, col] = last_valid
                
                # Backward fill for remaining NaN
                for i in range(len(col_data) - 1, -1, -1):
                    if not nan_mask[i]:
                        last_valid = col_data[i]
                    elif last_valid is not None:
                        data[i, col] = last_valid
        
        return data
    
    def _extract_windows(self, data: np.ndarray) -> tuple:
        """Extract windows from continuous data."""
        # Column indices for PAMAP2
        # 0: timestamp, 1: activity_id, 2: heart_rate
        # 3-19: hand IMU, 20-36: chest IMU, 37-53: ankle IMU
        
        # Select channels based on use_channels
        channel_indices = []
        
        if 'hand' in self.use_channels:
            # Hand IMU: columns 4-6 (acc), 10-12 (gyro), 13-15 (mag)
            channel_indices.extend([4, 5, 6, 10, 11, 12, 13, 14, 15])
        
        if 'chest' in self.use_channels:
            # Chest IMU: columns 21-23 (acc), 27-29 (gyro), 30-32 (mag)
            channel_indices.extend([21, 22, 23, 27, 28, 29, 30, 31, 32])
        
        if 'ankle' in self.use_channels:
            # Ankle IMU: columns 38-40 (acc), 44-46 (gyro), 47-49 (mag)
            channel_indices.extend([38, 39, 40, 44, 45, 46, 47, 48, 49])
        
        if self.use_heart_rate:
            channel_indices.append(2)
        
        channel_indices = sorted(channel_indices)
        
        # Extract sensor data and labels
        sensor_data = data[:, channel_indices]
        activity_labels = data[:, 1]  # Column 1 is activity ID
        
        # Filter to valid activities
        valid_mask = self._get_valid_activity_mask(activity_labels)
        
        windows = []
        labels = []
        boundaries = []
        
        # Create windows
        for i in range(0, len(sensor_data) - self.window_size, self.stride):
            window_data = sensor_data[i:i+self.window_size]
            window_labels = activity_labels[i:i+self.window_size]
            window_valid = valid_mask[i:i+self.window_size]
            
            # Skip if too many invalid samples
            if window_valid.sum() < self.window_size * 0.5:
                continue
            
            # Get majority label
            unique_labels, counts = np.unique(
                window_labels[window_valid], 
                return_counts=True
            )
            
            if len(unique_labels) == 0:
                continue
            
            majority_label = unique_labels[counts.argmax()]
            
            # Map to class index
            class_idx = self._activity_to_class_idx(majority_label)
            if class_idx is None:
                continue
            
            windows.append(window_data.T)  # [C, T]
            labels.append(class_idx)
            
            # Compute boundary score
            if len(unique_labels) > 1:
                boundary_score = 1.0 - (counts.max() / counts.sum())
            else:
                boundary_score = 0.0
            boundaries.append(boundary_score)
        
        return (
            np.array(windows),
            np.array(labels),
            np.array(boundaries)
        )
    
    def _get_valid_activity_mask(self, labels: np.ndarray) -> np.ndarray:
        """Get mask for valid activities."""
        valid_ids = set(self.ACTIVITY_LABELS.keys())
        
        if not self.include_optional:
            # Exclude optional protocol activities
            optional = {9, 10, 11, 18, 19, 20}
            valid_ids -= optional
        
        return np.isin(labels, list(valid_ids)) & (labels != 0)  # Exclude 'other'
    
    def _activity_to_class_idx(self, activity_id: int) -> Optional[int]:
        """Map activity ID to class index."""
        activity_map = {
            1: 0,   # lying
            2: 1,   # sitting
            3: 2,   # standing
            4: 3,   # walking
            5: 4,   # running
            6: 5,   # cycling
            7: 6,   # nordic_walking
            12: 7,  # ascending_stairs
            13: 8,  # descending_stairs
            16: 9,  # vacuum_cleaning
            17: 10, # ironing
            24: 11  # rope_jumping
        }
        return activity_map.get(activity_id)
    
    def _normalize_data(self):
        """Apply per-subject normalization."""
        for subject_id in torch.unique(self.subject_ids):
            mask = self.subject_ids == subject_id
            data = self.data[mask]
            
            mean = data.mean(dim=(0, 2), keepdim=True)
            std = data.std(dim=(0, 2), keepdim=True) + 1e-8
            
            self.data[mask] = (data - mean) / std
    
    def _download_dataset(self, root_path: Path):
        """Download PAMAP2 dataset."""
        import zipfile
        import urllib.request
        
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00231/PAMAP2_Dataset.zip"
        zip_path = root_path / "pamap2.zip"
        
        root_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Downloading PAMAP2 dataset from {url}...")
        urllib.request.urlretrieve(url, zip_path)
        
        print("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(root_path)
        
        zip_path.unlink()
        print("Download complete!")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'data': self.data[idx],
            'label': self.labels[idx],
            'boundary': self.boundaries[idx],
            'subject_id': self.subject_ids[idx],
            'dataset': 'pamap2'
        }
    
    @classmethod
    def get_loso_splits(cls, root: str, test_subject: int, **kwargs) -> Dict[str, 'PAMAP2Dataset']:
        """
        Get Leave-One-Subject-Out splits.
        
        Args:
            test_subject: Subject ID to use for testing (1-9)
        """
        all_subjects = list(range(1, 10))
        train_subjects = [s for s in all_subjects if s != test_subject]
        
        return {
            'train': cls(root=root, subjects=train_subjects, **kwargs),
            'test': cls(root=root, subjects=[test_subject], **kwargs)
        }
    
    def get_activity_type(self, label: int) -> str:
        """Get activity type (static/dynamic/transitional)."""
        if label in self.STATIC_ACTIVITIES:
            return 'static'
        elif label in self.DYNAMIC_ACTIVITIES:
            return 'dynamic'
        else:
            return 'transitional'
