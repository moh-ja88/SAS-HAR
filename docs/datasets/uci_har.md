# Dataset: UCI-HAR (UCI Human Activity Recognition)

## Overview

**Source:** University of California, Irvine  
**Year:** 2012  
**Citation:** Anguita et al., 2013

---

## Dataset Specifications

| Property | Value |
|----------|-------|
| **Subjects** | 30 |
| **Duration** | ~8 hours total |
| **Sampling Rate** | 50 Hz |
| **Modalities** | Accelerometer + Gyroscope (6 channels) |
| **Activities** | 6 |
| **Device** | Samsung Galaxy S II (waist-mounted) |

---

## Activities

| Activity | Description | Type |
|----------|-------------|------|
| WALKING | Normal walking | Dynamic |
| WALKING_UPSTAIRS | Walking up stairs | Dynamic |
| WALKING_DOWNSTAIRS | Walking down stairs | Dynamic |
| SITTING | Sedentary | Static |
| STANDING | Stationary | Static |
| LAYING | Lying down | Static |

---

## Data Format

### Raw Signals
```
- total_acc_x_train.txt, total_acc_y_train.txt, total_acc_z_train.txt
- body_gyro_x_train.txt, body_gyro_y_train.txt, body_gyro_z_train.txt
```

### Pre-computed Features
```
- 561 features in X_train.txt
- Labels in y_train.txt
- Subject IDs in subject_train.txt
```

---

## Preprocessing

### Standard Pipeline

1. **Noise filtering**: 3rd-order Butterworth low-pass (20 Hz cutoff)
2. **Gravity separation**: High-pass filter (0.3 Hz cutoff)
3. **Normalize**: [-1, 1] range
4. **Window**: 2.56 seconds (128 samples)
5. **Overlap**: 50%

### Code Example

```python
from scipy import signal

def preprocess_uci_har(raw_data, fs=50):
    # Butterworth low-pass filter
    b, a = signal.butter(3, 20/(fs/2), btype='low')
    filtered = signal.filtfilt(b, a, raw_data, axis=0)
    
    # High-pass filter for gravity separation
    b, a = signal.butter(3, 0.3/(fs/2), btype='high')
    body = signal.filtfilt(b, a, filtered, axis=0)
    
    # Normalize to [-1, 1]
    normalized = 2 * (body - body.min()) / (body.max() - body.min()) - 1
    
    return normalized
```

---

## Evaluation Protocol

### Standard Split (Fixed)
- **Train**: 21 subjects (70%)
- **Test**: 9 subjects (30%)
- Pre-defined split for reproducibility

---

## Benchmark Results

### State-of-the-Art

| Method | Year | Accuracy | F1-Score |
|--------|------|----------|----------|
| SVM + Features | 2013 | 89.3% | 87.5% |
| CNN | 2016 | 94.5% | 93.2% |
| LSTM | 2018 | 95.8% | 94.7% |
| Deep Similarity | 2024 | 95.1% | 94.7% |
| P2LHAP | 2024 | 96.5% | 95.8% |
| **SAS-HAR (Target)** | **2026** | **>98%** | **>97%** |

---

## Strengths

1. **Multi-modal** (Acc + Gyro)
2. **Higher sampling rate** (50 Hz)
3. **Pre-defined train/test split** for reproducibility
4. **Pre-computed features** available
5. **Standard benchmark** in HAR research

## Limitations

1. **Limited subjects** (30)
2. **No transitional activity labels**
3. **Controlled environment** (lab-based)
4. **Single device type**

---

## Download

**URL:** https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones

**License:** Publicly available for research

---

*Last Updated: March 2026*
