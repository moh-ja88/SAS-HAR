# Evaluation Metrics

## Overview

This document defines all evaluation metrics used in the SAS-HAR framework, including classification metrics, segmentation metrics, and efficiency metrics.

---

## 1. Classification Metrics

### 1.1 Accuracy

**Definition:** Ratio of correct predictions to total predictions.

**Formula:**
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**Use Case:** Overall performance when classes are balanced.

**Implementation:**
```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_true, y_pred)
```

---

### 1.2 Precision

**Definition:** Ratio of true positives to predicted positives.

**Formula:**
```
Precision = TP / (TP + FP)
```

**Use Case:** When false positives are costly.

---

### 1.3 Recall (Sensitivity)

**Definition:** Ratio of true positives to actual positives.

**Formula:**
```
Recall = TP / (TP + FN)
```

**Use Case:** When false negatives are costly (e.g., fall detection).

---

### 1.4 F1-Score

**Definition:** Harmonic mean of precision and recall.

**Formula:**
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**Use Case:** Balanced metric for imbalanced classes.

---

### 1.5 Weighted F1-Score

**Definition:** F1-score weighted by class support.

**Formula:**
```
Weighted F1 = Σ (F1_i × n_i) / N
```

where `n_i` is the number of samples in class i, and N is total samples.

**Use Case:** Primary classification metric for imbalanced HAR datasets.

**Implementation:**
```python
from sklearn.metrics import f1_score

weighted_f1 = f1_score(y_true, y_pred, average='weighted')
```

---

### 1.6 Confusion Matrix

**Definition:** Matrix showing predicted vs. actual labels.

**Use Case:** Identify which activities are confused with each other.

**Implementation:**
```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
```

---

## 2. Segmentation Metrics

### 2.1 Boundary F1-Score

**Definition:** F1-score for detecting activity boundaries.

**Formulation:**
- **True Positive (TP):** Predicted boundary within tolerance of ground truth
- **False Positive (FP):** Predicted boundary without corresponding ground truth
- **False Negative (FN):** Ground truth boundary not detected

**Tolerance:** Typically ±1 second (±20 samples at 20 Hz)

**Formula:**
```
Boundary Precision = TP_boundaries / (TP_boundaries + FP_boundaries)
Boundary Recall = TP_boundaries / (TP_boundaries + FN_boundaries)
Boundary F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**Implementation:**
```python
def boundary_f1_score(pred_boundaries, true_boundaries, tolerance=20):
    """
    Compute boundary F1-score
    
    Args:
        pred_boundaries: List of predicted boundary indices
        true_boundaries: List of ground truth boundary indices
        tolerance: Maximum distance (in samples) for a match
    
    Returns:
        precision, recall, f1
    """
    tp = 0
    matched_true = set()
    
    for pred_b in pred_boundaries:
        for true_b in true_boundaries:
            if abs(pred_b - true_b) <= tolerance and true_b not in matched_true:
                tp += 1
                matched_true.add(true_b)
                break
    
    fp = len(pred_boundaries) - tp
    fn = len(true_boundaries) - tp
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1
```

---

### 2.2 Segmentation Accuracy

**Definition:** Percentage of correctly classified segments.

**Formula:**
```
Segmentation Accuracy = Correctly Classified Segments / Total Segments
```

**Note:** A segment is correctly classified if its predicted label matches the majority ground truth label.

---

### 2.3 Hausdorff Distance

**Definition:** Maximum distance between predicted and true boundaries.

**Formula:**
```
Hausdorff Distance = max(
    max(min_distance(pred_b, true_boundaries)),
    max(min_distance(true_b, pred_boundaries))
)
```

**Use Case:** Measure boundary localization precision.

**Implementation:**
```python
def hausdorff_distance(pred_boundaries, true_boundaries):
    """
    Compute Hausdorff distance between boundary sets
    """
    def directed_hausdorff(set1, set2):
        return max(min(abs(b1 - b2) for b2 in set2) for b1 in set1)
    
    d1 = directed_hausdorff(pred_boundaries, true_boundaries)
    d2 = directed_hausdorff(true_boundaries, pred_boundaries)
    
    return max(d1, d2)
```

---

### 2.4 Rand Index

**Definition:** Measure of similarity between two segmentations.

**Formula:**
```
Rand Index = (TP + TN) / (N choose 2)
```

where:
- TP: Pairs in same segment in both
- TN: Pairs in different segments in both

---

### 2.5 Adjusted Rand Index (ARI)

**Definition:** Rand Index adjusted for chance.

**Formula:**
```
ARI = (RI - Expected_RI) / (Max_RI - Expected_RI)
```

**Use Case:** Standard metric for clustering/segmentation quality.

---

## 3. Transitional Activity Metrics

### 3.1 Transitional Activity F1

**Definition:** F1-score computed only for transitional activities.

**Transitional Activities:**
- Sit-to-Stand
- Stand-to-Sit
- Sit-to-Lie
- Lie-to-Sit
- Stand-to-Walk
- Walk-to-Stand

**Implementation:**
```python
def transitional_f1(y_true, y_pred, transitional_labels):
    """
    Compute F1-score for transitional activities only
    """
    # Filter for transitional activities
    mask = np.isin(y_true, transitional_labels)
    
    y_true_trans = y_true[mask]
    y_pred_trans = y_pred[mask]
    
    return f1_score(y_true_trans, y_pred_trans, average='weighted')
```

---

### 3.2 Transition Detection Rate

**Definition:** Percentage of transitions correctly detected.

**Formula:**
```
Detection Rate = Correctly Detected Transitions / Total Transitions
```

**Tolerance:** Transition detected if any sample within transition period is labeled as transition.

---

### 3.3 Transition Timing Error

**Definition:** Average error in transition onset detection.

**Formula:**
```
Timing Error = mean(|predicted_onset - true_onset|)
```

---

## 4. Efficiency Metrics

### 4.1 Model Size

**Definition:** Number of parameters in the model.

**Units:** Thousands (K) or Millions (M)

**Implementation:**
```python
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

print(f"Parameters: {count_parameters(model) / 1000:.1f}K")
```

---

### 4.2 Model Memory (Disk Size)

**Definition:** Size of saved model on disk.

**Units:** Kilobytes (KB) or Megabytes (MB)

**Implementation:**
```python
import os

# Save model
torch.save(model.state_dict(), 'model.pth')

# Get size
size_kb = os.path.getsize('model.pth') / 1024
print(f"Model size: {size_kb:.1f} KB")
```

---

### 4.3 FLOPs (Floating Point Operations)

**Definition:** Number of arithmetic operations per inference.

**Units:** Millions (M) or Billions (B)

**Implementation:**
```python
from fvcore.nn import FlopCountAnalysis

flops = FlopCountAnalysis(model, inputs)
print(f"FLOPs: {flops.total() / 1e6:.2f}M")
```

---

### 4.4 Inference Latency

**Definition:** Time for single inference.

**Units:** Milliseconds (ms)

**Implementation:**
```python
import time

# Warm-up
for _ in range(10):
    _ = model(inputs)

# Measure
start = time.perf_counter()
for _ in range(100):
    _ = model(inputs)
end = time.perf_counter()

latency_ms = (end - start) / 100 * 1000
print(f"Latency: {latency_ms:.2f} ms")
```

---

### 4.5 Energy Consumption

**Definition:** Energy consumed per inference.

**Units:** Nanojoules (nJ) or Microjoules (μJ)

**Measurement Methods:**

1. **Hardware Power Monitor:**
   - Measure current and voltage
   - Energy = Power × Time

2. **Software Estimation:**
   - FLOPs × Energy per FLOP
   - Platform-specific constants

**Implementation (Measurement):**
```python
# Using INA219 power sensor on Raspberry Pi
from ina219 import INA219

ina = INA219(shunt_ohms=0.1)

# Measure
start_time = time.time()
start_energy = ina.energy()

output = model(inputs)

end_time = time.time()
end_energy = ina.energy()

energy_nj = (end_energy - start_energy) * 1e9
latency_s = end_time - start_time
```

---

### 4.6 Peak Memory (RAM)

**Definition:** Maximum RAM usage during inference.

**Units:** Kilobytes (KB)

**Implementation:**
```python
import torch

torch.cuda.empty_cache() if torch.cuda.is_available() else None
torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None

output = model(inputs)

peak_memory_kb = torch.cuda.max_memory_allocated() / 1024 if torch.cuda.is_available() else 0
print(f"Peak Memory: {peak_memory_kb:.1f} KB")
```

---

### 4.7 Battery Life Impact

**Definition:** Estimated battery life reduction.

**Formula:**
```
Battery Life (hours) = Battery Capacity (mAh) / (Current Draw × Duty Cycle)
```

**Example:**
```
Battery: 200 mAh
Current Draw: 5 mA (during inference)
Duty Cycle: 1 inference per second = 0.001 (1ms/1000ms)

Average Current = 5 mA × 0.001 = 0.005 mA
Battery Life = 200 / 0.005 = 40,000 hours (4.5 years)
```

---

## 5. Summary Table

| Metric | Type | Formula | Target |
|--------|------|---------|--------|
| Accuracy | Classification | (TP+TN)/Total | >98% |
| Weighted F1 | Classification | Σ(F1_i × n_i)/N | >98% |
| **Boundary F1** | **Segmentation** | **2PR/(P+R)** | **>97%** |
| Segmentation Accuracy | Segmentation | Correct/Total | >95% |
| **Transitional F1** | **Transitional** | **F1 on transitions** | **>94%** |
| Parameters | Efficiency | Count | <25K |
| Model Size | Efficiency | Bytes | <100KB |
| FLOPs | Efficiency | Operations | <10M |
| Latency | Efficiency | Time | <1ms |
| **Energy** | **Efficiency** | **Joules** | **<45 nJ** |
| Memory | Efficiency | Bytes | <50KB |

---

## 6. Metric Selection Rationale

### Primary Metrics (Reported in Abstract)

1. **Accuracy** - Standard for HAR comparison
2. **Boundary F1** - Core segmentation metric
3. **Transitional F1** - Novel contribution focus
4. **Parameters** - Efficiency measure
5. **Energy** - Edge deployment metric

### Secondary Metrics (Reported in Tables)

- Precision, Recall per class
- Hausdorff distance
- FLOPs, Latency, Memory

---

*Last Updated: March 2026*
