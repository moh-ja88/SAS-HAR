# PAMAP2 Dataset

## Overview

The PAMAP2 (Physical Activity Monitoring in the Ageing Population) dataset is a comprehensive benchmark for HAR research, featuring 18 different physical activities collected from 9 subjects wearing multiple inertial measurement units (IMUs).

## Dataset Specifications

| Property | Value |
|----------|-------|
| **Subjects** | 9 (8 male, 1 female) |
| **Activities** | 18 types |
| **Sensors** | 3 IMUs (chest, wrist, ankle) + heart rate |
| **Sampling Rate** | 100 Hz |
| **Duration** | ~10 hours total |
| **Data Size** | ~2.5 GB |
| **Protocol** | Protocol 1 (all activities) and Protocol 2 (optional) |

## Sensor Configuration

### IMU Placement

```
┌─────────────────────────────────────────────┐
│           PAMAP2 Sensor Setup               │
├─────────────────────────────────────────────┤
│                                             │
│         [Chest IMU]                         │
│              │                              │
│              │                              │
│         ┌────┴────┐                         │
│         │  Heart  │                         │
│         │  Rate   │                         │
│         └─────────┘                         │
│                                             │
│     [Wrist IMU]                             │
│         │                                   │
│         │                                   │
│     [Ankle IMU]                             │
│                                             │
└─────────────────────────────────────────────┘
```

### IMU Channels (per sensor)

Each IMU provides 17 channels:
1. **Temperature** (1 channel)
2. **Accelerometer** (6 channels: ±16g and ±6g ranges)
3. **Gyroscope** (3 channels)
4. **Magnetometer** (3 channels)
5. **Orientation** (4 channels: quaternion)

**Total channels**: 3 IMUs × 17 + 1 heart rate = 52 channels

## Activity List

### Protocol 1 Activities (12 mandatory)

| ID | Activity | Type |
|----|----------|------|
| 1 | Lying | Static |
| 2 | Sitting | Static |
| 3 | Standing | Static |
| 4 | Walking | Dynamic |
| 5 | Running | Dynamic |
| 6 | Cycling | Dynamic |
| 7 | Nordic Walking | Dynamic |
| 12 | Ascending Stairs | Dynamic |
| 13 | Descending Stairs | Dynamic |
| 16 | Vacuum Cleaning | Dynamic |
| 17 | Ironing | Dynamic |
| 24 | Rope Jumping | Dynamic |

### Protocol 2 Activities (6 optional)

| ID | Activity | Type |
|----|----------|------|
| 9 | Watching TV | Static |
| 10 | Computer Work | Static |
| 11 | Car Driving | Dynamic |
| 18 | Folding Laundry | Dynamic |
| 19 | House Cleaning | Dynamic |
| 20 | Playing Soccer | Dynamic |

### Transitional/Optional Activities

| ID | Activity |
|----|----------|
| 0 | Other (transient activities) |
| 8 | Escalator Up/Down |

## Data Structure

### File Format

```
PAMAP2_Dataset/
├── Protocol/
│   ├── subject101.dat
│   ├── subject102.dat
│   ├── ...
│   └── subject109.dat
└── Optional/
    ├── subject101.dat
    └── ...
```

### Column Structure (52 columns)

```
Col 1:      timestamp
Col 2:      activity_id
Col 3:      heart_rate
Col 4-20:   IMU hand (chest)
Col 21-37:  IMU chest (wrist)  
Col 38-54:  IMU ankle
```

### Sample Data Loading

```python
import pandas as pd
import numpy as np

def load_pamap2(filepath):
    """
    Load PAMAP2 data file
    
    Args:
        filepath: path to .dat file
    
    Returns:
        DataFrame with labeled columns
    """
    columns = ['timestamp', 'activity_id', 'heart_rate']
    
    imu_cols = ['temp', 
                'acc_16g_x', 'acc_16g_y', 'acc_16g_z',
                'acc_6g_x', 'acc_6g_y', 'acc_6g_z',
                'gyro_x', 'gyro_y', 'gyro_z',
                'mag_x', 'mag_y', 'mag_z',
                'ori_w', 'ori_x', 'ori_y', 'ori_z']
    
    positions = ['hand', 'chest', 'ankle']
    
    for pos in positions:
        for col in imu_cols:
            columns.append(f'{pos}_{col}')
    
    df = pd.read_csv(filepath, sep=' ', header=None, names=columns)
    return df

# Example usage
df = load_pamap2('Protocol/subject101.dat')
print(df.shape)  # (num_samples, 54)
```

## Preprocessing

### Recommended Pipeline

```python
class PAMAP2Preprocessor:
    """
    Standard preprocessing for PAMAP2 dataset
    """
    def __init__(self, sampling_rate=100):
        self.sampling_rate = sampling_rate
        
    def preprocess(self, df, use_channels=None):
        """
        Apply standard preprocessing
        
        Args:
            df: raw DataFrame
            use_channels: list of channels to use (default: all IMU)
        
        Returns:
            preprocessed data and labels
        """
        # 1. Remove invalid activity IDs
        df = df[df['activity_id'] != 0]
        
        # 2. Interpolate missing heart rate values
        df['heart_rate'] = df['heart_rate'].interpolate(method='linear')
        
        # 3. Fill remaining NaN with forward fill
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # 4. Select channels
        if use_channels is None:
            # Use all IMU channels (exclude timestamp, activity, heart rate)
            imu_cols = [c for c in df.columns if any(
                pos in c for pos in ['hand', 'chest', 'ankle']
            ) and any(
                sensor in c for sensor in ['acc', 'gyro', 'mag']
            )]
            use_channels = imu_cols
        
        # 5. Normalize
        data = df[use_channels].values
        data = (data - data.mean()) / (data.std() + 1e-8)
        
        labels = df['activity_id'].values
        
        return data, labels
    
    def create_windows(self, data, labels, window_size=512, stride=256):
        """
        Create overlapping windows
        
        Args:
            data: (samples, channels) array
            labels: (samples,) array
            window_size: samples per window
            stride: step between windows
        
        Returns:
            windows: (num_windows, window_size, channels)
            window_labels: (num_windows,)
        """
        windows = []
        window_labels = []
        
        for i in range(0, len(data) - window_size, stride):
            window = data[i:i+window_size]
            label = np.bincount(labels[i:i+window_size].astype(int)).argmax()
            
            windows.append(window)
            window_labels.append(label)
        
        return np.array(windows), np.array(window_labels)
```

## Data Quality Notes

### Missing Data

- Heart rate has ~50% missing values (interpolation recommended)
- Some IMU channels have sporadic NaN values
- Activity 0 represents transient/unknown activities

### Known Issues

1. **Subject 109**: Only Protocol 1 data available
2. **Sensor dropouts**: Occasional brief signal losses
3. **Activity transitions**: Not explicitly labeled
4. **Class imbalance**: Some activities more frequent than others

## Class Distribution

```
Activity Distribution (Protocol 1):
─────────────────────────────────────────────────
Lying            ████████████████████ (normal)
Sitting          ████████████████████ (normal)
Standing         ████████████████████ (normal)
Walking          ████████████████████████████ (most)
Running          ██████████ (less)
Cycling          ████████████████ (normal)
Nordic Walking   ████████████████ (normal)
Ascending Stairs ████████ (less)
Descending Stairs████████ (less)
Vacuum Cleaning  ████████████ (normal)
Ironing          ████████████████ (normal)
Rope Jumping     ████ (least)
```

## Evaluation Protocol

### Standard Train/Test Split

```python
def get_pamap2_splits(subjects):
    """
    Leave-one-subject-out (LOSO) cross-validation
    
    Args:
        subjects: list of subject IDs [101, 102, ..., 109]
    
    Yields:
        train_subjects, test_subject for each fold
    """
    for test_subject in subjects:
        train_subjects = [s for s in subjects if s != test_subject]
        yield train_subjects, test_subject

# Example
for train_subs, test_sub in get_pamap2_splits(range(101, 110)):
    print(f"Train: {train_subs}, Test: {test_sub}")
```

### Recommended Metrics

| Metric | Purpose |
|--------|---------|
| Accuracy | Overall classification performance |
| Macro F1 | Balance across classes |
| Confusion Matrix | Per-class analysis |
| Subject-wise F1 | Cross-subject generalization |

## Benchmark Results

### Reported Performance (2020-2024)

| Method | Accuracy | Macro F1 | Protocol |
|--------|----------|----------|----------|
| DeepConvLSTM | 91.4% | 89.2% | LOSO |
| HAR-Transformer | 93.8% | 92.1% | LOSO |
| TinyHAR | 89.5% | 87.3% | LOSO |
| **SAS-HAR (target)** | **94.5%** | **93.0%** | LOSO |

### Segmentation Benchmarks

| Method | Boundary F1 | Mean Latency |
|--------|-------------|--------------|
| Sliding Window | 68.2% | 0ms |
| Adaptive Window | 74.5% | 50ms |
| Deep Similarity | 81.3% | 100ms |
| **SAS-HAR (target)** | **89.0%** | **80ms** |

## Usage in SAS-HAR

### Configuration

```python
pamap2_config = {
    'name': 'PAMAP2',
    'sampling_rate': 100,
    'window_size': 512,       # 5.12 seconds
    'stride': 256,            # 2.56 seconds overlap
    'channels': 27,           # 3 IMUs × 9 (acc+gyro+mag)
    'num_classes': 12,        # Protocol 1 activities
    'normalization': 'per_subject',
    'augmentation': ['jitter', 'scaling', 'rotation']
}
```

### Data Loader

```python
class PAMAP2DataLoader:
    def __init__(self, data_dir, config):
        self.data_dir = data_dir
        self.config = config
        self.preprocessor = PAMAP2Preprocessor(config['sampling_rate'])
    
    def load_subject(self, subject_id):
        filepath = f"{self.data_dir}/Protocol/subject{subject_id}.dat"
        df = load_pamap2(filepath)
        return self.preprocessor.preprocess(df)
    
    def get_dataloader(self, subjects, batch_size=32, shuffle=True):
        all_windows = []
        all_labels = []
        
        for subject_id in subjects:
            data, labels = self.load_subject(subject_id)
            windows, window_labels = self.preprocessor.create_windows(
                data, labels,
                self.config['window_size'],
                self.config['stride']
            )
            all_windows.append(windows)
            all_labels.append(window_labels)
        
        X = np.concatenate(all_windows)
        y = np.concatenate(all_labels)
        
        dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X),
            torch.LongTensor(y)
        )
        
        return torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle
        )
```

## References

1. Reiss, A., & Stricker, D. (2012). "Introducing a New Benchmarked Dataset for Activity Monitoring." *ISWC*.
2. Reiss, A., Stricker, D., & Hendeby, G. (2013). "Towards Application-Oriented Activity Recognition." *Biosignals*.
3. PAMAP2 Dataset: https://archive.ics.uci.edu/ml/datasets/pamap2+physical+activity+monitoring

## Download

```bash
# Download from UCI ML Repository
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00231/PAMAP2_Dataset.zip

# Or use our preprocessing script
python scripts/download_pamap2.py --output data/pamap2/
```
