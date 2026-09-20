"""
WISDM Activity Prediction Dataset

Wireless Sensor Data Mining dataset from:
https://archive.ics.uci.edu/ml/datasets/WISDM+Smartphone+and+Smartwatch+Activity+and+Biometrics+Dataset

Dataset characteristics:
- 36 subjects
- 6 activities: Walking, Jogging, Upstairs, Downstairs, Sitting, Standing
- 3-axial accelerometer at 20Hz
- Continuous time series (not pre-segmented)
"""

from pathlib import Path
from typing import Optional, List, Dict
import numpy as np
import torch
from sashar.data.base_dataset import BaseHARDataset


class WISDMDataset(BaseHARDataset):
    """WISDM Dataset loader for continuous HAR research."""
    
    NUM_CLASSES = 6
    SAMPLING_RATE = 20
    SENSOR_CHANNELS = {'accel': 3}
    ACTIVITY_LABELS = {
        0: 'Walking',
        1: 'Jogging',
        2: 'Upstairs',
        3: 'Downstairs',
        4: 'Sitting',
        5: 'Standing'
    }
    
    def __init__(
        self,
        root: str,
        window_size: int = 200,  # 10 seconds @ 20Hz
        stride: int = 100,   # 50% overlap
        split: str = 'train',
        subjects: Optional[List[int]] = None,
        use_norm: bool = True,
        download: bool = False,
        min_samples_per_activity: int = 10
    ):
        """
        Args:
            root: Root directory containing the dataset
            window_size: Window size in samples (default: 200 = 10s @ 20Hz)
            stride: Stride between windows (default: 100 = 50% overlap)
            split: One of 'train', 'test', 'val'
            subjects: Optional list of subject IDs to use
            use_norm: Whether to apply per-subject normalization
            download: Whether to download dataset if not found
            min_samples_per_activity: Minimum samples per activity for valid window
        """
        super().__init__(root, window_size, stride, split, subjects, use_norm)
        
        self.min_samples = min_samples_per_activity
        self._load_data()
    
    def _load_data(self):
        """Load and preprocess WISDM data."""
        root_path = Path(self.root)
        data_file = root_path / "WISDM_ar_v1.1" / "WISDM_ar_v1.1_raw.txt"
        
        if not data_file.exists():
            if self.download:
                self._download_dataset(root_path)
            else:
                raise FileNotFoundError(
                    f"WISDM dataset not found at {root_path}. "
                    f"Set download=True to automatically download."
                )
        
        # WISDM format: [user_id, activity, timestamp, x, y, z]
        # Read with pandas for efficiency
        import pandas as pd
        df = pd.read_csv(data_file, header=None, 
                        names=['user', 'activity', 'timestamp', 'x', 'y', 'z'],
                        comment='#')
        
        # Map activities to integers
        activity_map = {act: i for i, act in enumerate(self.ACTIVITY_LABELS.values())}
        df['label'] = df['activity'].map(activity_map)
        
        # Drop rows with unknown activities
        df = df.dropna(subset=['label'])
        df['label'] = df['label'].astype(int)
        
        # Filter by subjects if specified
        if self.subjects is not None:
            df = df[df['user'].isin(self.subjects)]
        
        # Create windows
        windows, labels, boundaries, user_ids = self._create_windows(df)
        
        # Convert to tensors
        self.data = torch.from_numpy(windows).float()
        self.labels = torch.from_numpy(labels).long()
        self.boundaries = torch.from_numpy(boundaries).float()
        self.subject_ids = torch.from_numpy(user_ids).long()
    
    def _create_windows(self, df) -> tuple:
        """Create sliding windows from continuous data."""
        windows = []
        labels = []
        boundaries = []
        user_ids = []
        
        # Group by user
        for user_id, user_df in df.groupby('user'):
            # Get sensor data and labels
            sensor_data = user_df[['x', 'y', 'z']].values
            activity_labels = user_df['label'].values
            
            # Normalize per subject if requested
            if self.use_norm:
                sensor_data = (sensor_data - sensor_data.mean(axis=0)) / (sensor_data.std(axis=0) + 1e-8)
            
            # Create windows
            for i in range(0, len(sensor_data) - self.window_size, self.stride):
                window = sensor_data[i:i+self.window_size]
                window_labels = activity_labels[i:i+self.window_size]
                
                # Check if window has enough samples
                unique_labels, counts = np.unique(window_labels, return_counts=True)
                
                if len(unique_labels) == 1:
                    # Single activity window
                    windows.append(window.T)  # [3, window_size]
                    labels.append(unique_labels[0])
                    boundaries.append(0.0)
                    user_ids.append(user_id)
                else:
                    # Transitional window - use majority label
                    majority_idx = counts.argmax()
                    majority_label = unique_labels[majority_idx]
                    
                    # Only include if majority is significant
                    if counts[majority_idx] / self.window_size >= 0.5:
                        windows.append(window.T)
                        labels.append(majority_label)
                        
                        # Mark as boundary if there's a significant activity change
                        boundary_score = 1.0 - (counts[majority_idx] / self.window_size)
                        boundaries.append(boundary_score)
                        user_ids.append(user_id)
        
        return (
            np.array(windows),
            np.array(labels),
            np.array(boundaries),
            np.array(user_ids)
        )
    
    def _download_dataset(self, root_path: Path):
        """Download WISDM dataset."""
        import zipfile
        import urllib.request
        
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00507/WISDM-ar-v1.1.zip"
        zip_path = root_path / "wisdm.zip"
        
        root_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Downloading WISDM dataset from {url}...")
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
            'dataset': 'wisdm'
        }
    
    @classmethod
    def get_standard_splits(cls, root: str, **kwargs) -> Dict[str, 'WISDMDataset']:
        """
        Get standard cross-subject splits for WISDM.
        
        Standard split (70/30):
        - Train: subjects 1-25
        - Test: subjects 26-36
        """
        return {
            'train': cls(root=root, subjects=list(range(1, 26)), **kwargs),
            'test': cls(root=root, subjects=list(range(26, 37)), **kwargs)
        }
    
    @staticmethod
    def create_boundary_labels(
        labels: np.ndarray,
        tolerance: int = 10
    ) -> np.ndarray:
        """
        Create boundary labels from continuous activity labels.
        
        Args:
            labels: Array of sample-level activity labels
            tolerance: Samples around boundary to mark as boundary
        
        Returns:
            Binary boundary labels
        """
        boundaries = np.zeros(len(labels), dtype=np.float32)
        
        # Find activity changes
        changes = np.where(labels[1:] != labels[:-1])[0] + 1
        
        # Mark boundary regions
        for change in changes:
            start = max(0, change - tolerance)
            end = min(len(labels), change + tolerance)
            boundaries[start:end] = 1.0
        
        return boundaries
