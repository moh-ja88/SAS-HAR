# Datasets Documentation

## Overview

This research uses four publicly available HAR benchmark datasets selected based on:
1. **Public availability** for reproducibility
2. **Multiple subjects** for cross-subject validation
3. **Transitional activities** for our specialized focus
4. **Multiple sensor modalities** for robust evaluation

---

## 1. WISDM Dataset

### Description
**Wireless Sensor Data Mining (WISDM)** - One of the most widely used HAR benchmarks for accelerometer-based activity recognition.

### Metadata

| Property | Value |
|----------|-------|
| **Year** | 2012 |
| **Source** | Fordham University |
| **Subjects** | 36 |
| **Duration** | 54 hours total |
| **Sampling Rate** | 20 Hz |
| **Modalities** | Accelerometer (3-axis) |
| **Activities** | 6 |
| **Device** | Android smartphone (waist) |

### Activities

| Activity | Description | Typical Duration |
|----------|-------------|------------------|
| Walking | Normal walking | Continuous |
| Jogging | Running | Continuous |
| Upstairs | Walking up stairs | Variable |
| Downstairs | Walking down stairs | Variable |
| Sitting | Sedentary | Extended |
| Standing | Stationary standing | Extended |

### Data Format

```
user_id,activity,timestamp,x_accel,y_accel,z_accel
```

### Preprocessing Steps

1. **Remove null values**
2. **Normalize per subject** (z-score)
3. **Window size**: 2.56 seconds (128 samples at 50 Hz resampled)
4. **Overlap**: 50%

### Evaluation Protocol

- **Cross-subject**: Train on 70% subjects, test on 30%
- **Stratified split** to maintain activity distribution
- **Report**: Accuracy, Weighted F1, Confusion Matrix

### References
- Kwapisz, J. R., Weiss, G. M., & Moore, S. A. (2011). Activity recognition using cell phone accelerometers. ACM SIGKDD Explorations.

---

## 2. UCI-HAR Dataset

### Description
**UCI Human Activity Recognition** - Standard benchmark with both accelerometer and gyroscope data from smartphone sensors.

### Metadata

| Property | Value |
|----------|-------|
| **Year** | 2012 |
| **Source** | UC Irvine |
| **Subjects** | 30 |
| **Duration** | ~8 hours total |
| **Sampling Rate** | 50 Hz |
| **Modalities** | Accelerometer + Gyroscope (6 channels) |
| **Activities** | 6 |
| **Device** | Samsung Galaxy S II (waist) |

### Activities

| Activity | Description | Type |
|----------|-------------|------|
| WALKING | Normal walking | Dynamic |
| WALKING_UPSTAIRS | Walking up stairs | Dynamic |
| WALKING_DOWNSTAIRS | Walking down stairs | Dynamic |
| SITTING | Sedentary | Static |
| STANDING | Stationary | Static |
| LAYING | Lying down | Static |

### Data Format

```
Raw signals:
- total_acc_x_train.txt, total_acc_y_train.txt, total_acc_z_train.txt
- body_gyro_x_train.txt, body_gyro_y_train.txt, body_gyro_z_train.txt

Pre-computed features:
- 561 features in X_train.txt
- Labels in y_train.txt
```

### Preprocessing Steps

1. **Noise filtering**: Butterworth low-pass (3rd order, 20 Hz cutoff)
2. **Gravity separation**: High-pass filter (0.3 Hz cutoff)
3. **Normalize**: [-1, 1] range
4. **Window**: 2.56 seconds (128 samples)
5. **Overlap**: 50%

### Evaluation Protocol

- **Fixed split**: 70% train (21 subjects), 30% test (9 subjects)
- **Standard protocol** for comparison with prior work

### References
- Anguita, D., et al. (2013). A public domain dataset for human activity recognition using smartphones. ESANN.

---

## 3. PAMAP2 Dataset

### Description
**Physical Activity Monitoring Protocol 2** - Rich dataset with 18 activities and multiple sensor modalities including heart rate.

### Metadata

| Property | Value |
|----------|-------|
| **Year** | 2013 |
| **Source** | University of Applied Sciences Upper Austria |
| **Subjects** | 9 |
| **Duration** | ~10 hours total |
| **Sampling Rate** | 100 Hz (IMU), 9 Hz (HR) |
| **Modalities** | Acc + Gyro + Mag + Heart Rate (54 channels) |
| **Activities** | 18 |
| **Device** | 3 IMUs (chest, dominant ankle, dominant wrist) |

### Activities

| ID | Activity | Type |
|----|----------|------|
| 1 | Lying | Static |
| 2 | Sitting | Static |
| 3 | Standing | Static |
| 4 | Walking | Dynamic |
| 5 | Running | Dynamic |
| 6 | Cycling | Dynamic |
| 7 | Nordic Walking | Dynamic |
| 9 | Watching TV | Static |
| 10 | Computer Work | Static |
| 11 | Car Driving | Static |
| 12 | Ascending Stairs | Dynamic |
| 13 | Descending Stairs | Dynamic |
| 16 | Vacuum Cleaning | Dynamic |
| 17 | Ironing | Dynamic |
| 18 | Folding Laundry | Dynamic |
| 19 | House Cleaning | Dynamic |
| 20 | Playing Soccer | Dynamic |
| 24 | Rope Jumping | Dynamic |

### Sensor Configuration

```
Chest IMU:
- Accelerometer (x, y, z)
- Gyroscope (x, y, z)
- Magnetometer (x, y, z)
- Orientation (quaternion: 4 values)

Dominant Wrist IMU:
- Same as above

Dominant Ankle IMU:
- Same as above

Heart Rate Monitor:
- BPM
```

### Preprocessing Steps

1. **Resample**: All to 100 Hz
2. **Interpolate**: Missing values
3. **Remove**: Invalid data (activity ID = 0)
4. **Normalize**: Per subject
5. **Window**: 5.12 seconds (512 samples)

### Evaluation Protocol

- **LOSO (Leave-One-Subject-Out)**: Train on 8, test on 1
- **Report**: Mean ± std across all folds

### References
- Reiss, A., & Stricker, D. (2012). Introducing a new benchmarked dataset for activity monitoring. ISWC.

---

## 4. Opportunity Dataset

### Description
**Opportunity** - Complex dataset with ambient and wearable sensors for challenging activity recognition scenarios.

### Metadata

| Property | Value |
|----------|-------|
| **Year** | 2010 |
| **Source** | EU Opportunity Project |
| **Subjects** | 4 |
| **Duration** | ~6 hours total |
| **Sampling Rate** | 30 Hz |
| **Modalities** | Multiple (wearable + ambient) |
| **Activities** | 5 (with sub-activities) |
| **Device** | Multiple IMUs + object sensors |

### Activities (High-Level)

| Activity | Description |
|----------|-------------|
| Relaxed | Standing, relaxed |
| Coffee | Making coffee |
| Sandwich | Preparing sandwich |
| Salad | Preparing salad |
| Cleanup | Cleaning up |

### Sensor Configuration

- **Body-worn**: 12 IMUs (arms, legs, back)
- **Object sensors**: 12 objects with accelerometers
- **Ambient**: 4 switches, 8 water sensors

### Preprocessing Steps

1. **Resample**: 30 Hz
2. **Select**: 113 features (as per standard protocol)
3. **Normalize**: [-1, 1]
4. **Window**: 1 second (30 samples)

### Evaluation Protocol

- **Run 1-2**: Training data
- **Run 3-5**: Test data
- **Metric**: Weighted accuracy

### References
- Chavarriaga, R., et al. (2013). The Opportunity challenge: A benchmark database for on-body sensor-based activity recognition. Pattern Recognition Letters.

---

## 5. Dataset Comparison

| Dataset | Activities | Subjects | Sensors | Best For |
|---------|-----------|----------|---------|----------|
| **WISDM** | 6 | 36 | Acc | Classification benchmark |
| **UCI-HAR** | 6 | 30 | Acc+Gyro | Standard benchmark |
| **PAMAP2** | 18 | 9 | Acc+Gyro+Mag+HR | Multi-modal, complex activities |
| **Opportunity** | 5+ | 4 | Multiple | Ambient sensing, fine-grained |

---

## 6. Custom Transitional Activity Dataset (Planned)

### Motivation
Existing datasets lack sufficient transitional activity labels for our specialized focus.

### Planned Collection

| Property | Target |
|----------|--------|
| **Subjects** | 20 |
| **Duration** | 2 hours each |
| **Activities** | Static + Transitions |
| **Sensors** | Acc + Gyro |
| **Labels** | Frame-level boundaries |

### Transitional Activities

1. Sit-to-Stand
2. Stand-to-Sit
3. Sit-to-Lie
4. Lie-to-Sit
5. Stand-to-Walk
6. Walk-to-Stand

### Annotation Protocol

- **Video synchronization** for ground truth
- **Frame-level boundaries** (sample-accurate)
- **Multiple annotators** (3) with agreement metric

---

## 7. Data Access

### Download Links

| Dataset | URL |
|---------|-----|
| WISDM | https://github.com/ShimmerEngineering/tinyos-shimmer/tree/main/apps/HAR |
| UCI-HAR | https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones |
| PAMAP2 | https://archive.ics.uci.edu/ml/datasets/pamap2+physical+activity+monitoring |
| Opportunity | https://archive.ics.uci.edu/ml/datasets/opportunity+activity+recognition |

### License

All datasets are publicly available for research purposes with appropriate citations.

---

*Last Updated: March 2026*
