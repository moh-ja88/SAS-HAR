# Individual Dataset Documentation

## WISDM Dataset

### Overview
- **Full Name:** Wireless Sensor Data Mining
- **Year:** 2012
- **Institution:** Fordham University
- **Public:** Yes

### Specifications
| Property | Value |
|----------|-------|
| Subjects | 36 |
| Activities | 6 |
| Duration | 54 hours |
| Sampling Rate | 20 Hz |
| Sensors | Accelerometer (3-axis) |
| Device | Android smartphone |
| Placement | Front pants pocket |

### Activities
1. Walking
2. Jogging
3. Upstairs
4. Downstairs
5. Sitting
6. Standing

### Preprocessing
- Normalize per subject
- Window size: 128 samples (6.4s at 20Hz)
- Overlap: 50%

### Evaluation Protocol
- Cross-subject: Train on 70%, test on 30%
- Stratified split

---

## UCI-HAR Dataset

### Overview
- **Full Name:** UCI Human Activity Recognition
- **Year:** 2012
- **Institution:** UC Irvine
- **Public:** Yes

### Specifications
| Property | Value |
|----------|-------|
| Subjects | 30 |
| Activities | 6 |
| Duration | 8 hours |
| Sampling Rate | 50 Hz |
| Sensors | Acc + Gyro (6 channels) |
| Device | Samsung Galaxy S II |
| Placement | Waist |

### Activities
1. WALKING
2. WALKING_UPSTAIRS
3. WALKING_DOWNSTAIRS
4. SITTING
5. STANDING
6. LAYING

### Preprocessing
- Butterworth low-pass filter
- Gravity separation (high-pass)
- Window size: 128 samples (2.56s)

### Evaluation Protocol
- Fixed split: 21 train / 9 test subjects

---

## PAMAP2 Dataset

### Overview
- **Full Name:** Physical Activity Monitoring Protocol 2
- **Year:** 2013
- **Institution:** University of Applied Sciences Upper Austria
- **Public:** Yes

### Specifications
| Property | Value |
|----------|-------|
| Subjects | 9 |
| Activities | 18 |
| Duration | 10 hours |
| Sampling Rate | 100 Hz |
| Sensors | Acc + Gyro + Mag + HR |
| Devices | 3 IMUs + HR monitor |
| Placement | Chest, wrist, ankle |

### Activities
1. Lying, 2. Sitting, 3. Standing, 4. Walking, 5. Running, 6. Cycling, 7. Nordic Walking, 9-24. Various household and exercise activities

### Preprocessing
- Resample to 100 Hz
- Remove invalid data
- Window size: 512 samples (5.12s)

### Evaluation Protocol
- LOSO (Leave-One-Subject-Out)

---

## Opportunity Dataset

### Overview
- **Full Name:** Opportunity Activity Recognition
- **Year:** 2010
- **Institution:** EU Opportunity Project
- **Public:** Yes

### Specifications
| Property | Value |
|----------|-------|
| Subjects | 4 |
| Activities | 5 (with sub-activities) |
| Duration | 6 hours |
| Sampling Rate | 30 Hz |
| Sensors | Multiple wearable + ambient |
| Devices | 12 IMUs + object sensors |

### Activities
1. Relaxed
2. Coffee (making)
3. Sandwich (preparing)
4. Salad (preparing)
5. Cleanup

### Preprocessing
- Select 113 features
- Normalize to [-1, 1]
- Window size: 30 samples (1s)

### Evaluation Protocol
- Runs 1-2: Training
- Runs 3-5: Testing

---

*Last Updated: March 2026*
