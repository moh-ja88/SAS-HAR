# Opportunity Dataset

## Overview

The Opportunity dataset is a challenging benchmark for human activity recognition, featuring complex activities of daily living (ADL) in a realistic home environment setting. It is known for its high sensor density and detailed annotations.

## Dataset Specifications

| Property | Value |
|----------|-------|
| **Subjects** | 4 (3 male, 1 female) |
| **Activities** | 17 + drill activities |
| **Sensors** | 72+ sensors (IMUs, accelerometers, object sensors) |
| **Sampling Rate** | 30 Hz (standardized) |
| **Duration** | ~6 hours total |
| **Environment** | Simulated apartment |
| **Annotations** | Frame-level activity labels |

## Sensor Configuration

### Body-Worn Sensors

```
┌─────────────────────────────────────────────────────────────┐
│           Opportunity Sensor Setup                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  [IMU - Back]                    [IMU - RUA*]               │
│       │                               │                     │
│  [IMU - LUA*]                    [IMU - RLA*]               │
│       │                               │                     │
│  [IMU - LLA*]                    [IMU - R-Shoe]             │
│       │                                                     │
│  [IMU - L-Shoe]                                             │
│                                                             │
│  *RUA = Right Upper Arm, LLA = Left Lower Arm, etc.         │
│                                                             │
│  Total: 7 body-worn IMUs (36 inertial channels)             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Object Sensors

12 objects instrumented with 3D accelerometers:
- Door (fridge, dishwasher, etc.)
- Drawer (multiple)
- Cups
- Spoon
- Knife
- And more...

### Ambient Sensors

- 12 switches on doors/drawers
- Environmental sensors

### Total Sensor Count

| Type | Count | Channels |
|------|-------|----------|
| Body IMUs | 7 | 36 (acc + gyro + mag) |
| Object Accel | 12 | 36 |
| Ambient | 12+ | 12+ |
| **Total** | **31+** | **84+** |

## Activity Annotations

### High-Level Activities (17 classes)

| ID | Activity | Description |
|----|----------|-------------|
| 1 | Open Door | Opening various doors |
| 2 | Open Dishwasher | Opening dishwasher |
| 3 | Close Dishwasher | Closing dishwasher |
| 4 | Open Fridge | Opening refrigerator |
| 5 | Close Fridge | Closing refrigerator |
| 6 | Open Drawer 1 | Lower drawer |
| 7 | Close Drawer 1 | Lower drawer |
| 8 | Open Drawer 2 | Middle drawer |
| 9 | Close Drawer 2 | Middle drawer |
| 10 | Open Drawer 3 | Upper drawer |
| 11 | Close Drawer 3 | Upper drawer |
| 12 | Clean Table | Wiping table surface |
| 13 | Drink from Cup | Using instrumented cup |
| 14 | Toggle Switch | Light switches |
| 15 | Null | No specific activity |

### Drill Activities

Additional scripted sequence including:
- Walking
- Sitting
- Standing
- Lying down

### Annotation Granularity

- **Frame-level**: 30 Hz annotations
- **Mode labels**: "execute", "reach", "relax"
- **Null class**: Significant portion (~40%)

## Data Structure

### File Format

```
OpportunityDataset/
├── S1-ADL1.dat        # Subject 1, ADL run 1
├── S1-ADL2.dat
├── S1-ADL3.dat
├── S1-ADL4.dat
├── S1-ADL5.dat
├── S1-Drill.dat       # Subject 1, drill sequence
├── S2-ADL1.dat
├── ...
└── S4-Drill.dat
```

### Column Structure (250 columns)

```
Columns 1-5:     Sensor annotations (timestamps, labels)
Columns 6-113:   Inertial sensors (accelerometer, gyro, magnetometer)
Columns 114-131: Object acceleration sensors
Columns 132-242: Additional sensor modalities
Columns 243-250: Activity labels (locomotion, gestures)
```

### Sample Data Loading

```python
import pandas as pd
import numpy as np

# Column names for Opportunity
COLUMN_NAMES = [
    # Time and metadata
    'time', 'sample', 'run',
    # Inertial sensors
    *[f'IMU_{i}_{axis}' for i in range(7) for axis in ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']],
    # Object sensors
    *[f'OBJ_{i}_{axis}' for i in range(12) for axis in ['x', 'y', 'z']],
    # Labels
    'locomotion', 'gesture_left', 'gesture_right'
]

def load_opportunity(filepath):
    """
    Load Opportunity data file
    
    Args:
        filepath: path to .dat file
    
    Returns:
        DataFrame with data
    """
    df = pd.read_csv(filepath, sep=' ', header=None)
    # Select relevant columns based on your needs
    return df
```

## Preprocessing

### Standard Pipeline

```python
class OpportunityPreprocessor:
    """
    Preprocessing for Opportunity dataset
    """
    def __init__(self, sampling_rate=30):
        self.sampling_rate = sampling_rate
        
    def preprocess(self, df, use_modalities='body'):
        """
        Preprocess Opportunity data
        
        Args:
            df: raw DataFrame
            use_modalities: 'body', 'object', 'all'
        
        Returns:
            data, labels
        """
        # 1. Select modalities
        if use_modalities == 'body':
            # Body-worn IMU columns (typically columns 6-113)
            sensor_cols = list(range(5, 113))
        elif use_modalities == 'all':
            sensor_cols = list(range(5, 242))
        else:
            sensor_cols = list(range(5, 113))
        
        # 2. Handle missing values
        data = df[sensor_cols].values
        data = np.nan_to_num(data, nan=0.0)
        
        # 3. Normalize (per-channel)
        data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)
        
        # 4. Get labels
        labels = df.iloc[:, 243].values  # Locomotion label
        
        # 5. Remove null class if desired
        valid_mask = labels > 0
        data = data[valid_mask]
        labels = labels[valid_mask]
        
        return data, labels
    
    def create_windows(self, data, labels, window_size=128, stride=64):
        """
        Create overlapping windows
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

### Challenges

1. **High dimensionality**: 250+ columns require careful selection
2. **Missing data**: Significant NaN values in some channels
3. **Class imbalance**: Null class dominates (~40%)
4. **Subject variability**: Only 4 subjects
5. **Complex annotations**: Multiple label types per frame

### Sensor Failures

Some runs have known sensor issues:
- S2-ADL3: Missing hand IMU data
- Some object sensors occasionally fail

## Class Distribution

```
Opportunity Activity Distribution (excluding Null):
────────────────────────────────────────────────────
Open Fridge       ████████
Close Fridge      ████████
Open Dishwasher   ██████
Close Dishwasher  ██████
Open Drawer 1     █████
Close Drawer 1    █████
Open Drawer 2     █████
Close Drawer 2    █████
Open Drawer 3     ████
Close Drawer 3    ████
Open Door         ████████
Clean Table       ██████
Drink from Cup    █████
Toggle Switch     ████
────────────────────────────────────────────────────
Null Class        ████████████████████████████████████ (~40%)
```

## Evaluation Protocol

### Standard Splits

```python
# Standard Opportunity evaluation (ADL runs)
TRAIN_RUNS = ['S1-ADL1', 'S1-ADL2', 'S1-ADL3', 'S2-ADL1', 'S2-ADL2',
              'S3-ADL1', 'S3-ADL2', 'S4-ADL1', 'S4-ADL2']

TEST_RUNS = ['S1-ADL4', 'S1-ADL5', 'S2-ADL3', 'S2-ADL4',
             'S3-ADL3', 'S3-ADL4', 'S4-ADL2', 'S4-ADL3']

# Alternative: Leave-one-subject-out
def get_loso_splits():
    subjects = [1, 2, 3, 4]
    for test_sub in subjects:
        train_subs = [s for s in subjects if s != test_sub]
        yield train_subs, test_sub
```

### Metrics

| Metric | Purpose |
|--------|---------|
| Accuracy | Overall (including null) |
| Accuracy (no null) | Excluding null class |
| F1-score | Per-class balance |
| Confusion Matrix | Error analysis |

## Benchmark Results

### Reported Performance (2013-2024)

| Method | Accuracy | Accuracy (no null) | F1 |
|--------|----------|-------------------|-----|
| Baseline (Chavarriaga) | 64.4% | 72.1% | 0.59 |
| DeepConvLSTM | 71.3% | 79.8% | 0.68 |
| Attention-HAR | 74.5% | 82.3% | 0.71 |
| **SAS-HAR (target)** | **76.0%** | **84.0%** | **0.74** |

### Opportunity Challenge (2011)

Official competition results established baselines for:
- Gesture recognition
- Locomotion detection
- Null detection

## Usage in SAS-HAR

### Configuration

```python
opportunity_config = {
    'name': 'Opportunity',
    'sampling_rate': 30,
    'window_size': 128,       # ~4.3 seconds
    'stride': 64,             # 50% overlap
    'channels': 113,          # Body IMU sensors
    'num_classes': 17,        # High-level activities
    'include_null': False,    # Exclude null class
    'normalization': 'per_run',
    'augmentation': ['jitter', 'time_warp']
}
```

### Data Loader

```python
class OpportunityDataLoader:
    def __init__(self, data_dir, config):
        self.data_dir = data_dir
        self.config = config
        self.preprocessor = OpportunityPreprocessor(config['sampling_rate'])
    
    def load_run(self, subject_id, run_id):
        filepath = f"{self.data_dir}/S{subject_id}-ADL{run_id}.dat"
        df = load_opportunity(filepath)
        return self.preprocessor.preprocess(
            df, 
            use_modalities='body'
        )
    
    def get_dataloader(self, runs, batch_size=32, shuffle=True):
        all_windows = []
        all_labels = []
        
        for run in runs:
            # Parse run name (e.g., 'S1-ADL1')
            subject_id = int(run[1])
            run_id = int(run.split('-ADL')[1])
            
            data, labels = self.load_run(subject_id, run_id)
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

## Special Considerations

### For Segmentation Research

Opportunity is particularly useful for:
1. **Fine-grained transitions**: Many object interactions
2. **Multi-sensor fusion**: Body + object + ambient
3. **Realistic scenarios**: Actual daily living activities

### Challenges for SAS-HAR

1. **High sensor count**: May require channel selection
2. **Short activities**: Some interactions are brief
3. **Low sampling rate**: 30 Hz limits temporal resolution
4. **Few subjects**: Generalization concerns

## References

1. Chavarriaga, R., et al. (2013). "The Opportunity Challenge: A Benchmark Database for On-Body Sensor-Based Activity Recognition." *Pattern Recognition Letters*.
2. Roggen, D., et al. (2010). "Collecting Complex Activity Datasets in Highly Rich Networked Sensor Environments." *INSS*.
3. Opportunity Dataset: https://archive.ics.uci.edu/ml/datasets/opportunity+activity+recognition

## Download

```bash
# Download from UCI ML Repository
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00226/OpportunityUCIDataset.zip

# Or use our preprocessing script
python scripts/download_opportunity.py --output data/opportunity/
```

## Related Datasets

- **Opportunity++**: Extended version with more subjects
- **HCI-Tagging**: Human-computer interaction activities
- **RHAD**: Richmond Human Activity Dataset (similar setup)
