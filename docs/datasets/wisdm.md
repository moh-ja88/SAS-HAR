# Dataset: WISDM (Wireless Sensor Data Mining)

## Overview

**Source:** Fordham University  
**Year:** 2012  
**Citation:** Kwapisz et al., 2011

---

## Dataset Specifications

| Property | Value |
|----------|-------|
| **Subjects** | 36 |
| **Duration** | 54 hours total |
| **Sampling Rate** | 20 Hz |
| **Modalities** | Accelerometer (3-axis) |
| **Activities** | 6 |
| **Device** | Android smartphone (waist-mounted) |

---

## Activities

| Activity | Description | Typical Duration |
|----------|-------------|------------------|
| Walking | Normal walking | Continuous |
| Jogging | Running | Continuous |
| Upstairs | Walking up stairs | Variable |
| Downstairs | Walking down stairs | Variable |
| Sitting | Sedentary | Extended |
| Standing | Stationary | Extended |

---

## Data Format

```
user_id,activity,timestamp,x_accel,y_accel,z_accel
```

### Sample Data
```
33,Walking,42444083373511,-0.5,9.8,1.3
33,Walking,42444083373521,-0.5,9.8,1.2
33,Walking,42444083373531,-0.4,9.9,1.3
```

---

## Preprocessing

### Standard Pipeline

1. **Remove null values**
2. **Normalize per subject** (z-score: mean=0, std=1)
3. **Window size**: 2.56 seconds (128 samples at 50 Hz resampled)
4. **Overlap**: 50%

### Code Example

```python
import pandas as pd
import numpy as np

def preprocess_wisdm(filepath):
    # Load data
    columns = ['user', 'activity', 'timestamp', 'x', 'y', 'z']
    df = pd.read_csv(filepath, header=None, names=columns)
    
    # Remove nulls
    df = df.dropna()
    
    # Normalize per user
    for user in df['user'].unique():
        mask = df['user'] == user
        for col in ['x', 'y', 'z']:
            df.loc[mask, col] = (df.loc[mask, col] - df.loc[mask, col].mean()) / df.loc[mask, col].std()
    
    return df
```

---

## Evaluation Protocol

### Standard Split
- **Train**: 70% of subjects (25 subjects)
- **Test**: 30% of subjects (11 subjects)
- **Stratified** to maintain activity distribution

### Cross-Validation
- **Leave-One-Subject-Out (LOSO)**: Train on 35, test on 1
- Report mean ± std across all folds

---

## Benchmark Results

### State-of-the-Art

| Method | Year | Accuracy | F1-Score |
|--------|------|----------|----------|
| FSW + CNN | 2016 | 91.2% | 89.8% |
| FSW + LSTM | 2018 | 93.5% | 92.1% |
| Deep Similarity | 2024 | 94.2% | 93.8% |
| P2LHAP | 2024 | 95.8% | 95.2% |
| **SAS-HAR (Target)** | **2026** | **>98%** | **>97%** |

---

## Strengths

1. **Large subject pool** (36) for generalization testing
2. **Simple sensor modality** (accelerometer only)
3. **Well-established benchmark** for comparison
4. **Publicly available**

## Limitations

1. **Single modality** (no gyroscope)
2. **Low sampling rate** (20 Hz)
3. **Limited transitional activities**
4. **No boundary labels** (activity labels only)

---

## Download

**URL:** https://github.com/ShimmerEngineering/tinyos-shimmer/tree/main/apps/HAR

**License:** Publicly available for research

---

*Last Updated: March 2026*
