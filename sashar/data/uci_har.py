"""
UCI-HAR Dataset Loader

UCI Human Activity Recognition dataset from:
https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones

Dataset characteristics:
- 30 subjects (age 19-48 years)
- 6 activities: WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING
- 3-axial accelerometer and 3-axial gyroscope at 50Hz
- Pre-segmented into 2.56s windows with 50% overlap
"""

from pathlib import Path
from typing import Optional, List, Dict, Tuple
import numpy as np
import torch
from sashar.data.base_dataset import BaseHARDataset


class UCIHardataset(BaseHARDataset):
    """UCI-HAR Dataset loader."""
    
    NUM_CLASSES = 6
    SAMPLING_RATE = 50
    SENSOR_CHANNELS = {'accel': 3, 'gyro': 3}
    ACTIVITY_LABELS = {
        0: 'WALKING',
        1: 'WALKING_UPSTAIRS',
        2: 'WALKING_DOWNSTAIRS',
        3: 'SITTING',
        4: 'STANDING',
        5: 'LAYING'
    }
    
    def __init__(
        self,
        root: str,
        window_size: int = 128,
        stride: int = 64,
        split: str = 'train',
        subjects: Optional[List[int]] = None,
        use_norm: bool = True,
        download: bool = False
    ):
        """
        Args:
            root: Root directory containing the dataset
            window_size: Window size in samples (default: 128 = 2.56s @ 50Hz)
            stride: Stride between windows (default: 64 = 50% overlap)
            split: One of 'train', 'test', 'val'
            subjects: Optional list of subject IDs to use (for cross-subject)
            use_norm: Whether to apply normalization
            download: Whether to download dataset if not found
        """
        super().__init__(root, window_size, stride, split, subjects, use_norm)
        
        self.split = split
        self.use_norm = use_norm
        
        # Load data
        self._load_data()
    
    def _load_data(self):
        """Load and preprocess UCI-HAR data."""
        root_path = Path(self.root)
        
        # Check if data exists
        train_file = root_path / "train" / "X_train.txt"
        test_file = root_path / "test" / "X_test.txt"
        train_labels = root_path / "train" / "y_train.txt"
        test_labels = root_path / "test" / "y_test.txt"
        
        if not train_file.exists():
            if self.download:
                self._download_dataset(root_path)
            else:
                raise FileNotFoundError(
                    f"UCI-HAR dataset not found at {root_path}. "
                    f"Set download=True to automatically download."
                )
        
        # Load based on split
        if self.split == 'train':
            X = np.loadtxt(train_file)
            y = np.loadtxt(train_labels)
        elif self.split == 'test':
            X = np.loadtxt(test_file)
            y = np.loadtxt(test_labels)
        else:
            # Combine for validation
            X_train = np.loadtxt(train_file)
            X_test = np.loadtxt(test_file)
            y_train = np.loadtxt(train_labels)
            y_test = np.loadtxt(test_labels)
            X = np.vstack([X_train, X_test])
            y = np.concatenate([y_train, y_test])
        
        # Reshape: UCI-HAR provides flattened 561 features per window
        # Original format: [561 features] = [128 timesteps * 9 channels - 1]
        # We reshape to [samples, channels, time]
        n_samples = X.shape[0]
        X = X[:, :-1]  # Remove last column (subject ID in some versions)
        X = X.reshape(n_samples, 128, 9)  # [N, 128, 9]
        X = X.transpose(0, 2, 1)  # [N, 9, 128]
        
        # Labels are 1-indexed in UCI-HAR
        y = y - 1
        
        # Filter by subjects if specified
        if self.subjects is not None:
            # Subject IDs are embedded in the data
            subject_file = root_path / "train" / "subject_train.txt"
            if subject_file.exists():
                subjects_arr = np.loadtxt(subject_file) - 1  # 1-indexed
                mask = np.isin(subjects_arr, self.subjects)
                X = X[mask]
                y = y[mask]
        
        # Normalize if requested
        if self.use_norm:
            X = (X - X.mean()) / (X.std() + 1e-8)
        
        self.data = torch.from_numpy(X).float()
        self.labels = torch.from_numpy(y).long()
    
    def _download_dataset(self, root_path: Path):
        """Download UCI-HAR dataset."""
        import zipfile
        import urllib.request
        
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
        zip_path = root_path / "uci_har.zip"
        
        root_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Downloading UCI-HAR dataset from {url}...")
        urllib.request.urlretrieve(url, zip_path)
        
        print("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(root_path)
        
        # Move files to expected locations
        extracted = root_path / "UCI HAR Dataset"
        if extracted.exists():
            import shutil
            shutil.move(str(extracted / "train"), str(root_path / "train"))
            shutil.move(str(extracted / "test"), str(root_path / "test"))
            shutil.rmtree(extracted)
        
        zip_path.unlink()
        print("Download complete!")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'data': self.data[idx],
            'label': self.labels[idx],
            'subject_id': torch.tensor(0),  # UCI-HAR doesn't provide per-sample subject
            'dataset': 'uci_har'
        }
    
    @classmethod
    def get_subject_split(cls, subjects: List[int], **kwargs) -> 'UCIHardataset':
        """Create dataset split for specific subjects."""
        return cls(subjects=subjects, **kwargs)
    
    @staticmethod
    def create_boundary_labels(
        labels: np.ndarray,
        tolerance: int = 10
    ) -> np.ndarray:
        """
        Create boundary labels from activity labels.
        
        In UCI-HAR, each window has one label, so we create synthetic boundaries
        where activity changes between consecutive windows.
        
        Args:
            labels: Array of window labels
            tolerance: Not used for UCI-HAR (pre-segmented)
        
        Returns:
            Binary boundary labels (all zeros for UCI-HAR)
        """
        # UCI-HAR is pre-segmented, no boundaries within windows
        return np.zeros(len(labels), dtype=np.float32)
    
    def get_cross_subject_splits(self) -> Dict[str, List[int]]:
        """
        Get standard cross-subject evaluation splits.
        
        UCI-HAR standard split:
        - Train: subjects 1-21 (70%)
        - Test: subjects 22-30 (30%)
        """
        return {
            'train': list(range(1, 22)),
            'test': list(range(22, 31))
        }


if __name__ == "__main__":
    # Test dataset loading
    dataset = UCIHardataset(root="data/uci_har", split="train", download=True)
    print(f"Loaded {len(dataset)} training samples")
    
    stats = dataset.get_statistics()
    print(f"Dataset statistics: {stats}")
