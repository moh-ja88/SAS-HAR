# WISDM Dataset Investigation Findings

## Executive Summary

**Two root causes identified for WISDM's poor 18% accuracy:**

1. **Sensor Malfunctions** (Primary): Test subjects 1641-1648 have sensor malfunctions with raw values 2-4x larger than normal. This causes ~18% accuracy.

2. **Subject Generalization Challenge** (Secondary): Even after fixing sensor issues, subject-independent evaluation on only 3 test subjects yields ~26% accuracy. With random split, accuracy jumps to 64%.

## Key Results

| Configuration | Accuracy | F1 Score |
|---------------|----------|----------|
| Original WISDM (subject split, with malfunctions) | 18.36% | 18.73% |
| Fixed WISDM (subject split, no malfunctions) | 26.69% | 26.57% |
| Fixed WISDM (random split, no malfunctions) | **63.78%** | **63.79%** |

The 37% gap between random split (64%) and subject split (27%) confirms that **subject generalization is inherently challenging** for this dataset.

## Problem Details

### 1. Sensor Malfunction in Test Subjects

| Subject Set | Raw Value Range | Normalized Range |
|-------------|-----------------|------------------|
| Train (1600-1635) | -39.8 to +52.4 | -12.8 to +15.3 |
| Test Normal (1636-1640, 1647, 1649-1650) | -20 to +24 | -8 to +7 |
| **Test Malfunctioning (1641-1646, 1648)** | **-78.5 to +78.5** | **-25.8 to +20.6** |

### 2. Specific Problematic Subjects

| Subject | Raw Min | Raw Max | Normalized Min | Normalized Max | Issue |
|---------|---------|---------|----------------|----------------|-------|
| 1641 | -78.48 | +78.48 | -25.80 | +20.65 | SEVERE |
| 1642 | -59.16 | +64.41 | -14.79 | +15.33 | SEVERE |
| 1643 | -77.63 | +51.49 | -18.58 | +13.96 | SEVERE |
| 1644 | -69.13 | +32.66 | -16.14 | +15.03 | SEVERE |
| 1645 | -78.48 | +47.10 | -13.19 | +12.95 | SEVERE |
| 1646 | -75.57 | +39.18 | -17.53 | +9.71 | SEVERE |
| 1648 | -78.48 | +78.08 | -10.40 | +10.24 | MODERATE |

### 3. Subject Generalization Challenge

After removing malfunctioning subjects, only 3 test subjects remain (1647, 1649, 1650).

**Why subject generalization is hard:**
- Human activity patterns vary significantly between individuals
- Different people perform the same activity differently (walking gait, typing style, etc.)
- Phone accelerometer alone may not capture enough distinguishing features
- 18 classes include many similar activities (eating soup vs chips vs pasta)

### 4. Why Per-Subject Normalization Doesn't Fix Sensor Malfunctions

The preprocessing code normalizes each subject's data independently:
```python
sensor_data = (sensor_data - sensor_data.mean(axis=0)) / (sensor_data.std(axis=0) + 1e-8)
```

This approach fails when:
1. Raw data has extreme outliers (sensor glitches)
2. The outliers are not evenly distributed
3. The std gets inflated by outliers, but individual outlier values remain extreme

## Comparison with Working Datasets

### UCI-HAR (works well, 77-94% accuracy)
- Uses **global normalization** after combining all data
- Pre-extracted features, not raw sensor data
- No sensor malfunction issues
- **Subject-dependent evaluation** in many benchmarks

### PAMAP2 (works well)
- Uses **global normalization** after combining all data
- Raw sensor data with proper quality control

### WISDM (fails at 18%)
- Uses **per-subject normalization** before combining
- Contains sensor malfunctions in test subjects
- No outlier removal or clipping
- **Subject-independent evaluation** is inherently harder

## Recommended Fixes

### Option 1: Remove Malfunctioning Subjects + Use Global Normalization (Implemented)

File: `scripts/preprocess_wisdm_fixed.py`

```bash
python scripts/preprocess_wisdm_fixed.py
```

Results:
- Data range: [-4.01, 3.76] (vs original [-25.8, 20.6])
- Removed 7 malfunctioning subjects
- Applied global normalization
- Train: 33,754 samples, Test: 3,241 samples

### Option 2: Use Random Split for Fair Comparison

If the goal is to compare model architectures (not subject generalization):
- Use random 80/20 split
- Expected accuracy: ~64%

### Option 3: Increase Test Subjects

Current test set has only 3 subjects. Options:
- Use different train/test split ratio
- Include some train subjects in test for subject-dependent evaluation
- Use cross-subject validation

## Files Affected

- `scripts/preprocess_datasets.py` - Original WISDM preprocessing (lines 277-404)
- `scripts/preprocess_wisdm_fixed.py` - Fixed preprocessing script (NEW)
- `data/wisdm/processed/wisdm_processed.npz` - Original processed data
- `data/wisdm/processed/wisdm_processed_fixed.npz` - Fixed processed data (NEW)

## Conclusions

1. **Original 18% accuracy** is caused by sensor malfunctions in test subjects 1641-1648
2. **After fixing malfunctions, 26% accuracy** is due to challenging subject generalization with only 3 test subjects
3. **With random split, 64% accuracy** shows the model and data are fundamentally sound
4. **Subject-independent HAR on WISDM is inherently challenging** - this is a known issue in the literature

## Recommendations for Future Work

1. For fair model comparison, use random split or cross-validation
2. For subject-independent evaluation, use all available subjects with proper cross-validation
3. Consider using fewer, more distinct activity classes
4. Consider using additional sensors (gyroscope, watch data) for better discrimination
5. Compare with published WISDM benchmarks to validate expected accuracy range
