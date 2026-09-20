"""
Opportunity Dataset for Human Activity Recognition

A comprehensive dataset loader for the Opportunity dataset, one of the most
challenging HAR benchmarks with complex activities of daily living.

Dataset Details:
- 4 subjects performing activities of daily living
- 17+ high-level activities (fine-grained) + 4 locomotion modes
- 72+ sensors (body-worn IMUs, object sensors, ambient)
- 30 Hz sampling rate
- Realistic home environment setup
- Standard train/test protocol defined by dataset authors

URL: https://archive.ics.uci.edu/ml/datasets/OPPORTUNIST+Activity+Recognition+Challenge

Citation:
@article{chavarriaga2013opportunity,
  title={The Opportunity challenge: A benchmark database for on-body sensor-based activity recognition},
  author={Chavarriaga, Ricardo and Bayati, Hesam and Mill{'a}n, Jos{'e} del R},
  journal={Pattern Recognition Letters},
  volume={34},
  number={15},
  pages={2033--2042},
  year={2013},
  publisher={Elsevier}
}
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import requests
from numpy.typing import NDArray
from torch.utils.data import DataLoader

from .base_dataset import BaseHARDataset


class OpportunityDataset(BaseHARDataset):
    """
    Opportunity Dataset Loader.
    
    The Opportunity dataset is a challenging benchmark for human activity
    recognition in a realistic home environment.
    """
    
    SAMPLING_RATE: int = 30
    
    SENSOR_CHANNELS: dict[str, int] = {
        'imu_back': 9,
        'imu_rua': 9,
        'imu_rla': 9,
        'imu_lua': 9,
        'imu_lla': 9,
        'imu_lshoe': 9,
        'imu_rshoe': 9,
        'object_sensors': 12,
        'ambient': 5
    }
    
    ACTIVITY_LABELS: dict[int, str] = {
        0: 'null',
        1: 'open_door',
        2: 'open_dishwasher',
        3: 'close_dishwasher',
        4: 'open_fridge',
        5: 'close_fridge',
        6: 'open_drawer1',
        7: 'close_drawer1',
        8: 'open_drawer2',
        9: 'close_drawer2',
        10: 'open_drawer3',
        11: 'close_drawer3',
        12: 'clean_table',
        13: 'drink_from_cup',
        14: 'toggle_switch',
        15: 'null2',
        16: 'lie_down',
        17: 'stand_up'
    }
    
    LOCOMOTION_LABELS: dict[int, str] = {
        0: 'null',
        1: 'stand',
        2: 'walk',
        3: 'sit',
        4: 'lie'
    }
    
    STATIC_ACTIVITIES: list[int] = [1, 3, 5, 7, 9, 11, 12, 14, 16]
    DYNAMIC_ACTIVITIES: list[int] = [2, 4, 6, 8, 10, 13, 17]
    TRANSITIONAL_ACTIVITIES: list[int] = [16, 17]
    
    SUBJECTS: list[int] = [1, 2, 3, 4]
    
    DATASET_URL: str = "https://archive.ics.uci.edu/ml/machine-learning-databases/00226/OpportunityUCIDataset.zip"
    
    def __init__(
        self,
        root: str,
        window_size: int = 60,
        stride: int | None = None,
        subjects: list[int] | None = None,
        include_boundaries: bool = True,
        transform: object | None = None,
        download: bool = False,
        label_type: str = 'high_level',
        sensor_config: str = 'body_worn',
        include_null: bool = False,
        split: str | None = None
    ):
        """
        Initialize Opportunity HAR dataset.
        
        Args:
            root: Root directory for data storage
            window_size: Number of samples per window (default: 60 = 2s @ 30Hz)
            stride: Stride between windows (default: window_size // 2)
            subjects: List of subject IDs to include (None = all)
            include_boundaries: Whether to include boundary labels
            transform: Optional transforms to apply
            download: Whether to download data if not present
            label_type: 'high_level' for 17 activities, 'locomotion' for 4 modes
            sensor_config: 'body_worn', 'all', or 'minimal'
            include_null: Include null class in training
            split: 'train', 'test', or None (all data)
        """
        self.label_type = label_type
        self.sensor_config = sensor_config
        self.include_null = include_null
        self.split = split
        
        # Set NUM_CLASSES based on label type (override class attribute)
        if label_type == 'locomotion':
            object.__setattr__(self, 'NUM_CLASSES', 4)
        else:
            object.__setattr__(self, 'NUM_CLASSES', 17)
        
        # Store subjects list
        if subjects is not None:
            self._subjects_list = [s for s in subjects if s in self.SUBJECTS]
            if not self._subjects_list:
                raise ValueError("No valid subjects specified")
        else:
            self._subjects_list = self.SUBJECTS
        
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
        """Total number of sensor channels based on config."""
        # Must match _get_sensor_indices() column selection
        if self.sensor_config == 'minimal':
            return 27  # range(3, 30)
        elif self.sensor_config == 'body_worn':
            return 110  # range(3, 113)
        else:
            return 240  # range(3, 243) - all sensors
    
    def _check_data_exists(self) -> bool:
        """Check if processed data exists on disk."""
        processed_path = self.root / 'processed' / f'opportunity_{self.sensor_config}_{self.label_type}.npz'
        return processed_path.exists()
    
    def _load_data(self) -> tuple[NDArray[np.float32], NDArray[np.int64], NDArray[np.float32] | None, NDArray[np.int64]]:
        """Load Opportunity HAR data from disk."""
        processed_path = self.root / 'processed'
        data_file = processed_path / f'opportunity_{self.sensor_config}_{self.label_type}.npz'
        
        if not data_file.exists():
            raise FileNotFoundError(
                f"Processed data not found at {data_file}. "
                "Call download_and_preprocess() first."
            )
        
        loaded = np.load(data_file, allow_pickle=True)
        
        data = loaded['data'].astype(np.float32)
        labels = loaded['labels'].astype(np.int64)
        subject_ids = loaded['subject_ids'].astype(np.int64)
        boundaries = loaded.get('boundaries', None)
        if boundaries is not None:
            boundaries = boundaries.astype(np.float32)
        
        # Filter by split if specified
        if self.split is not None:
            split_info = loaded.get('split_info', None)
            if split_info is not None:
                split_mask = split_info == self.split
                data = data[split_mask]
                labels = labels[split_mask]
                subject_ids = subject_ids[split_mask]
                if boundaries is not None:
                    boundaries = boundaries[split_mask]
        
        # Filter by subjects
        if self._subjects_list != self.SUBJECTS:
            subject_mask = np.isin(subject_ids, np.array(self._subjects_list))
            data = data[subject_mask]
            labels = labels[subject_mask]
            subject_ids = subject_ids[subject_mask]
            if boundaries is not None:
                boundaries = boundaries[subject_mask]
        
        # Filter null class if needed
        if not self.include_null:
            valid_mask = labels >= 0
            data = data[valid_mask]
            labels = labels[valid_mask]
            subject_ids = subject_ids[valid_mask]
            if boundaries is not None:
                boundaries = boundaries[valid_mask]
        
        return data, labels, boundaries, subject_ids
    
    def _create_windows(self) -> tuple[NDArray[np.float32], NDArray[np.int64], NDArray[np.float32] | None]:
        """
        Create windows from data.
        
        For Opportunity, data is already pre-windowed by download_and_preprocess(),
        so we return it directly rather than re-windowing.
        
        Returns:
            windows: Windowed data [M, C, window_size]
            labels: Window labels [M]
            boundaries: Window boundaries [M, 1] or None (expanded from scalar)
        """
        # Data is already windowed [N, C, T] from preprocessing
        # Boundaries are scalar per-window scores, expand to [N, 1] for compatibility
        if self.boundaries is not None:
            # Expand scalar boundaries [N] -> [N, 1] for base class compatibility
            boundaries_expanded = self.boundaries[:, np.newaxis]
        else:
            boundaries_expanded = None
        
        return self.data, self.labels, boundaries_expanded
    
    @classmethod
    def download_and_preprocess(
        cls,
        root: str,
        sensor_config: str = 'body_worn',
        label_type: str = 'high_level'
    ) -> None:
        """Download and preprocess Opportunity dataset."""
        root_path = Path(root)
        raw_dir = root_path / 'raw'
        processed_dir = root_path / 'processed'
        
        raw_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        cls._download_raw_data(raw_dir)
        
        print("Processing Opportunity dataset...")
        
        train_runs: list[str] = []
        test_runs: list[str] = []
        
        for subject in cls.SUBJECTS:
            train_runs.extend([
                f'S{subject}-ADL1', f'S{subject}-ADL2', f'S{subject}-ADL3', f'S{subject}-Drill'
            ])
            test_runs.extend([
                f'S{subject}-ADL4', f'S{subject}-ADL5'
            ])
        
        all_data: list[NDArray[np.float32]] = []
        all_labels: list[NDArray[np.int64]] = []
        all_subjects: list[int] = []
        all_boundaries: list[float] = []
        all_splits: list[str] = []
        
        for run_name in train_runs + test_runs:
            split = 'train' if run_name in train_runs else 'test'
            subject_id = int(run_name[1])
            
            run_data = cls._load_run_data(raw_dir, run_name)
            
            if run_data is None:
                print(f"Warning: Could not load {run_name}")
                continue
            
            windows, labels, boundaries = cls._extract_windows_from_run(
                run_data,
                sensor_config=sensor_config,
                label_type=label_type
            )
            
            if len(windows) == 0:
                continue
            
            all_data.append(windows)
            all_labels.append(labels)
            all_boundaries.extend(boundaries)
            all_subjects.extend([subject_id] * len(windows))
            all_splits.extend([split] * len(windows))
        
        if not all_data:
            raise ValueError("No data was successfully loaded!")
        
        data = np.concatenate(all_data, axis=0).astype(np.float32)
        labels = np.concatenate(all_labels, axis=0).astype(np.int64)
        subject_ids_arr = np.array(all_subjects, dtype=np.int64)
        boundaries_arr = np.array(all_boundaries, dtype=np.float32) if all_boundaries else None
        split_info = np.array(all_splits)
        
        data = cls._normalize_data(data)
        
        output_file = processed_dir / f'opportunity_{sensor_config}_{label_type}.npz'
        save_dict: dict[str, Any] = {
            'data': data,
            'labels': labels,
            'subject_ids': subject_ids_arr,
            'split_info': split_info
        }
        if boundaries_arr is not None:
            save_dict['boundaries'] = boundaries_arr
        np.savez(output_file, **save_dict)
        
        print(f"Processed data saved to {output_file}")
        print(f"Total samples: {len(data)}")
        print(f"Shape: {data.shape}")
        print(f"Train samples: {(split_info == 'train').sum()}")
        print(f"Test samples: {(split_info == 'test').sum()}")
    
    @classmethod
    def _download_raw_data(cls, raw_dir: Path) -> None:
        """Download raw Opportunity data."""
        if (raw_dir / 'download_complete').exists():
            print("Raw data already downloaded.")
            return
        
        zip_path = raw_dir / 'OpportunityUCIDataset.zip'
        
        try:
            print(f"Downloading from {cls.DATASET_URL}...")
            response = requests.get(cls.DATASET_URL, stream=True, timeout=60)
            response.raise_for_status()
            
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print("Extracting...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(raw_dir)
            
            (raw_dir / 'download_complete').touch()
            zip_path.unlink()
            
            print("Download complete!")
            
        except requests.RequestException as e:
            print(f"Download failed: {e}")
            print("Please download manually from:")
            print("https://archive.ics.uci.edu/ml/datasets/OPPORTUNIST+Activity+Recognition+Challenge")
            raise
    
    @classmethod
    def _load_run_data(cls, raw_dir: Path, run_name: str) -> NDArray[np.float32] | None:
        """Load a single run's data file."""
        data_path = raw_dir / 'OpportunityUCIDataset' / 'dataset'
        
        possible_files = [
            data_path / f'{run_name}.dat',
            data_path / f'{run_name}.csv',
            data_path / f'{run_name.lower()}.dat',
        ]
        
        run_file = None
        for f in possible_files:
            if f.exists():
                run_file = f
                break
        
        if run_file is None:
            return None
        
        try:
            data = np.loadtxt(run_file, dtype=np.float32)
            return data
        except Exception as e:
            print(f"Error loading {run_file}: {e}")
            return None
    
    @classmethod
    def _extract_windows_from_run(
        cls,
        data: NDArray[np.float32],
        sensor_config: str,
        label_type: str,
        window_size: int = 60,
        stride: int = 30
    ) -> tuple[NDArray[np.float32], NDArray[np.int64], list[float]]:
        """Extract windows from a single run's data."""
        sensor_indices = cls._get_sensor_indices(sensor_config)
        label_col = 243 if label_type == 'high_level' else 244
        
        sensor_data = data[:, sensor_indices]
        sensor_data = np.nan_to_num(sensor_data, nan=0.0)
        
        labels = data[:, label_col]
        valid_label_mask = ~np.isnan(labels)
        
        windows: list[NDArray[np.float32]] = []
        window_labels: list[int] = []
        boundaries: list[float] = []
        
        for i in range(0, len(data) - window_size, stride):
            window_data = sensor_data[i:i + window_size]
            window_labels_raw = labels[i:i + window_size]
            
            valid_in_window = valid_label_mask[i:i + window_size]
            if valid_in_window.sum() < window_size * 0.5:
                continue
            
            valid_labels = window_labels_raw[valid_in_window]
            if len(valid_labels) == 0:
                continue
            
            unique_labels, counts = np.unique(valid_labels, return_counts=True)
            majority_label = unique_labels[counts.argmax()]
            
            class_idx = cls._label_to_class_idx(float(majority_label), label_type)
            if class_idx is None:
                continue
            
            windows.append(window_data.T)
            window_labels.append(class_idx)
            
            if len(unique_labels) > 1:
                boundary_score = float(1.0 - (counts.max() / counts.sum()))
            else:
                boundary_score = 0.0
            boundaries.append(boundary_score)
        
        if not windows:
            empty_data = np.array([], dtype=np.float32).reshape(0, len(sensor_indices), window_size)
            empty_labels = np.array([], dtype=np.int64)
            return empty_data, empty_labels, []
        
        return (
            np.stack(windows).astype(np.float32),
            np.array(window_labels, dtype=np.int64),
            boundaries
        )
    
    @classmethod
    def _get_sensor_indices(cls, sensor_config: str) -> list[int]:
        """Get column indices for selected sensor configuration."""
        if sensor_config == 'minimal':
            return list(range(3, 30))
        elif sensor_config == 'body_worn':
            return list(range(3, 113))
        else:
            return list(range(3, 243))
    
    @classmethod
    def _label_to_class_idx(cls, label: float, label_type: str) -> int | None:
        """Map raw label value to class index."""
        label_int = int(label)
        
        if label_type == 'high_level':
            if label_int < 1 or label_int > 17:
                return None
            return label_int - 1
        else:
            if label_int < 1 or label_int > 4:
                return None
            return label_int - 1
    
    @classmethod
    def _normalize_data(cls, data: NDArray[np.float32]) -> NDArray[np.float32]:
        """Apply z-score normalization per channel."""
        mean = np.mean(data, axis=(0, 2), keepdims=True)
        std = np.std(data, axis=(0, 2), keepdims=True) + 1e-8
        return ((data - mean) / std).astype(np.float32)
    
    def get_subject_split(
        self,
        train_subjects: list[int] | None = None,
        val_subjects: list[int] | None = None,
        test_subjects: list[int] | None = None
    ) -> tuple[OpportunityDataset | None, OpportunityDataset | None, OpportunityDataset | None]:
        """Create subject-based data splits."""
        train_ds: OpportunityDataset | None = None
        val_ds: OpportunityDataset | None = None
        test_ds: OpportunityDataset | None = None
        
        if train_subjects is not None:
            train_ds = OpportunityDataset(
                root=str(self.root),
                window_size=self.window_size,
                stride=self.stride,
                subjects=train_subjects,
                label_type=self.label_type,
                sensor_config=self.sensor_config,
                include_null=self.include_null,
                split=self.split
            )
        
        if val_subjects is not None:
            val_ds = OpportunityDataset(
                root=str(self.root),
                window_size=self.window_size,
                stride=self.stride,
                subjects=val_subjects,
                label_type=self.label_type,
                sensor_config=self.sensor_config,
                include_null=self.include_null,
                split=self.split
            )
        
        if test_subjects is not None:
            test_ds = OpportunityDataset(
                root=str(self.root),
                window_size=self.window_size,
                stride=self.stride,
                subjects=test_subjects,
                label_type=self.label_type,
                sensor_config=self.sensor_config,
                include_null=self.include_null,
                split=self.split
            )
        
        return train_ds, val_ds, test_ds
    
    def get_standard_splits(self) -> tuple[OpportunityDataset, OpportunityDataset]:
        """Get the standard Opportunity train/test splits."""
        train_ds = OpportunityDataset(
            root=str(self.root),
            window_size=self.window_size,
            stride=self.stride,
            subjects=self._subjects_list,
            label_type=self.label_type,
            sensor_config=self.sensor_config,
            include_null=self.include_null,
            split='train'
        )
        
        test_ds = OpportunityDataset(
            root=str(self.root),
            window_size=self.window_size,
            stride=self.stride,
            subjects=self._subjects_list,
            label_type=self.label_type,
            sensor_config=self.sensor_config,
            include_null=self.include_null,
            split='test'
        )
        
        return train_ds, test_ds
    
    def get_activity_distribution(self) -> dict[str, float]:
        """Get distribution of activities in dataset."""
        counts = np.bincount(self.window_labels, minlength=self.NUM_CLASSES)
        total = counts.sum()
        
        if self.label_type == 'high_level':
            labels = self.ACTIVITY_LABELS
        else:
            labels = self.LOCOMOTION_LABELS
        
        return {
            labels.get(i + 1, f"class_{i}"): count / total
            for i, count in enumerate(counts)
            if count > 0
        }
    
    def get_statistics(self) -> dict[str, Any]:
        """Get dataset statistics for reproducibility documentation."""
        stats = super().get_statistics()
        stats.update({
            'label_type': self.label_type,
            'sensor_config': self.sensor_config,
            'include_null': self.include_null,
            'split': self.split,
            'num_subjects': len(np.unique(self.subject_ids)),
        })
        return stats


def create_opportunity_loaders(
    root: str,
    sensor_config: str = 'body_worn',
    label_type: str = 'high_level',
    batch_size: int = 64,
    **kwargs: Any
) -> tuple[DataLoader[Any], DataLoader[Any]]:
    """
    Create train/test dataloaders for Opportunity HAR.
    
    Args:
        root: Root directory for data
        sensor_config: 'body_worn', 'all', or 'minimal'
        label_type: 'high_level' or 'locomotion'
        batch_size: Batch size for all loaders
        **kwargs: Additional arguments passed to dataset
    
    Returns:
        Tuple of (train_loader, test_loader)
    """
    dataset = OpportunityDataset(
        root=root,
        sensor_config=sensor_config,
        label_type=label_type,
        download=True,
        **kwargs
    )
    
    train_ds, test_ds = dataset.get_standard_splits()
    
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=4
    )
    
    return train_loader, test_loader


DATASET_INFO: dict[str, Any] = {
    'name': 'Opportunity Activity Recognition',
    'abbreviation': 'OPP',
    'year': 2013,
    'subjects': 4,
    'activities': {
        'high_level': 17,
        'locomotion': 4
    },
    'sensors': 72,
    'sampling_rate': 30,
    'duration_hours': 6,
    'url': 'https://archive.ics.uci.edu/ml/datasets/OPPORTUNIST+Activity+Recognition+Challenge',
    'license': 'Public',
    'citation': '''
@article{chavarriaga2013opportunity,
  title={The Opportunity challenge: A benchmark database for on-body sensor-based activity recognition},
  author={Chavarriaga, Ricardo and Bayati, Hesam and Mill{\\'a}n, Jos{\\'e} del R},
  journal={Pattern Recognition Letters},
  volume={34},
  number={15},
  pages={2033--2042},
  year={2013},
  publisher={Elsevier}
}
'''
}
