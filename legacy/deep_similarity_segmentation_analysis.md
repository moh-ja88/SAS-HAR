# Core Paper Analysis: Deep Similarity Segmentation Model for Sensor-Based Activity Recognition

## Bibliographic Information

- **Title:** Deep Similarity Segmentation Model for Sensor-Based Activity Recognition
- **Authors:** AbdulRahman Baraka, Mohd Halim Mohd Noor
- **Year:** 2024
- **Venue:** Multimedia Tools and Applications (Springer)
- **DOI:** 10.1007/s11042-024-18933-2

---

## 1. Problem Definition

### Core Problem
Fixed-size sliding window (FSW) segmentation creates semantic ambiguity by:
1. Cutting across activity boundaries
2. Capturing multiple activities in one window
3. Losing contextual information at boundaries

### Limitations of Previous Work (Statistical Similarity)
- Hand-crafted statistical features
- Limited representational power
- Similarity metrics not optimized for HAR
- No learning of optimal features

### Goal
Develop a deep learning approach that learns optimal similarity representations for boundary detection.

---

## 2. Core Method

### Algorithm Overview

```
Deep Similarity Segmentation Pipeline:
1. Input: Continuous sensor stream S
2. Window Extraction: Extract overlapping windows W₁, ..., Wₙ
3. Feature Learning:
   a. Pass each window through shared CNN encoder
   b. Extract learned feature representations F₁, ..., Fₙ
4. Similarity Network:
   a. Form window pairs (Wᵢ, Wᵢ₊₁)
   b. Concatenate features: [Fᵢ; Fᵢ₊₁]
   c. Pass through similarity network
   d. Output: Probability of boundary
5. Boundary Detection: Apply threshold
6. Output: Variable-length segments
```

### Key Innovation
Replace statistical features with **learned representations** via CNNs, and treat segmentation as **binary classification** (same activity vs. boundary).

---

## 3. Mathematical Formulation

### 3.1 CNN Feature Encoder

```
Feature Extraction:
F = f_CNN(W; θ)

where:
- W = input window [C × T] (C channels, T timesteps)
- f_CNN = convolutional neural network
- θ = learned parameters
- F = feature vector [d-dimensional]
```

### 3.2 CNN Architecture

```
f_CNN Architecture:
Input: W ∈ ℝ^(C×T)
  ↓
Conv1D Layer 1: 64 filters, kernel=5, stride=1
  + BatchNorm + ReLU + MaxPool(2)
  ↓
Conv1D Layer 2: 128 filters, kernel=3, stride=1
  + BatchNorm + ReLU + MaxPool(2)
  ↓
Conv1D Layer 3: 256 filters, kernel=3, stride=1
  + BatchNorm + ReLU
  ↓
Global Average Pooling
  ↓
Output: F ∈ ℝ^256
```

### 3.3 Similarity Network

```
Similarity Computation:
P(boundary|Wᵢ, Wᵢ₊₁) = σ(g_sim([Fᵢ; Fᵢ₊₁; Fᵢ⊙Fᵢ₊₁; |Fᵢ-Fᵢ₊₁|]))

where:
- [;] = concatenation
- ⊙ = element-wise multiplication
- g_sim = multi-layer perceptron (MLP)
- σ = sigmoid activation
```

### 3.4 Similarity Features

Four types of features combined:
1. **Fᵢ:** Features from first window
2. **Fᵢ₊₁:** Features from second window
3. **Fᵢ ⊙ Fᵢ₊₁:** Element-wise product (similarity)
4. **|Fᵢ - Fᵢ₊₁|:** Absolute difference (dissimilarity)

### 3.5 Loss Function

```
Binary Cross-Entropy Loss:
L = -1/N Σᵢ [yᵢ log(pᵢ) + (1-yᵢ) log(1-pᵢ)]

where:
- yᵢ ∈ {0, 1} = ground truth label
  - 0 = same activity (no boundary)
  - 1 = boundary exists
- pᵢ = predicted boundary probability
- N = number of window pairs
```

---

## 4. Model Architecture

### Complete Architecture

```
Deep Similarity Segmentation Model:
├── Shared CNN Encoder (Siamese-like)
│   ├── Conv1D Block 1: 64 filters, kernel=5
│   ├── Conv1D Block 2: 128 filters, kernel=3
│   ├── Conv1D Block 3: 256 filters, kernel=3
│   └── Global Average Pooling
│
├── Feature Combination Layer
│   ├── Concatenate: [Fᵢ; Fᵢ₊₁]
│   ├── Element-wise product: Fᵢ⊙Fᵢ₊₁
│   ├── Absolute difference: |Fᵢ-Fᵢ₊₁|
│   └── Final concatenation → 1024-dim vector
│
├── Similarity Network (MLP)
│   ├── Dense 512 + ReLU + Dropout(0.5)
│   ├── Dense 256 + ReLU + Dropout(0.5)
│   ├── Dense 128 + ReLU
│   └── Dense 1 + Sigmoid
│
└── Boundary Detection
    ├── Threshold comparison (τ = 0.5)
    ├── Segment formation
    └── Output: Variable-length segments
```

### Parameter Count
- CNN Encoder: ~450K parameters
- Similarity Network: ~200K parameters
- **Total: ~650K parameters**

---

## 5. Training Procedure

### 5.1 Ground Truth Generation

```
Boundary Labels:
- Use activity labels to identify true boundaries
- For each adjacent window pair (Wᵢ, Wᵢ₊₁):
  - If activity(Wᵢ) ≠ activity(Wᵢ₊₁):
    → y = 1 (boundary)
  - Else:
    → y = 0 (same activity)
```

### 5.2 Data Augmentation

```
Augmentation Techniques:
1. Random window shifts: ±10% of window size
2. Gaussian noise injection: σ = 0.01
3. Time warping: ±5% speed variation
4. Sensor dropout: Randomly zero 10% of channels
```

### 5.3 Training Configuration

```
Optimizer: Adam
Learning Rate: 0.001
Batch Size: 64 window pairs
Epochs: 100
Early Stopping: Patience = 10
Validation Split: 20%
```

---

## 6. Strengths

### 6.1 Learned Representations
- **Advantage:** Features optimized for segmentation task
- **Comparison:** Statistical features (Baraka 2023) are hand-crafted
- **Impact:** Better capture of complex temporal patterns

### 6.2 End-to-End Trainable
- **Advantage:** Joint optimization of encoder and similarity network
- **Impact:** Task-specific feature learning
- **Benefit:** No manual feature engineering

### 6.3 Improved Performance
- **Quantitative:** +5-8% over statistical approach
- **Qualitative:** Better transitional activity detection
- **Robustness:** More robust to sensor noise

### 6.4 Flexible Architecture
- **Modularity:** Can use different CNN backbones
- **Adaptability:** Works with different sensor modalities
- **Compatibility:** Downstream classifier agnostic

---

## 7. Weaknesses

### 7.1 Increased Complexity
- **Issue:** 650K parameters vs. statistical (negligible)
- **Impact:** Higher memory and computational requirements
- **Challenge:** Not suitable for resource-constrained devices

### 7.2 Still Local Context
- **Issue:** Only compares adjacent windows
- **Limitation:** No long-range dependency modeling
- **Impact:** May miss patterns spanning multiple windows

### 7.3 Training Data Dependency
- **Issue:** Requires labeled activity boundaries
- **Cost:** Expensive annotation process
- **Limitation:** May not generalize to unseen activity types

### 7.4 Hyperparameter Sensitivity
- **Sensitivity to:**
  - Network architecture choices
  - Learning rate, batch size
  - Dropout rates
  - Threshold τ selection
- **Impact:** Requires careful tuning per dataset

### 7.5 No Edge Optimization
- **Issue:** Model too large for wearable devices
- **Gap:** No quantization, pruning, or distillation
- **Deployment:** Cloud/server inference required

---

## 8. Performance Results

### 8.1 Benchmark Results

| Dataset | Accuracy | F1-Score | Transitional F1 | Improvement over Statistical |
|---------|----------|----------|-----------------|------------------------------|
| WISDM | 94.2% | 93.8% | 87.3% | +5.1% |
| UCI-HAR | 95.1% | 94.7% | 88.9% | +6.3% |
| PAMAP2 | 92.8% | 92.1% | 84.6% | +7.2% |

### 8.2 Computational Metrics

| Metric | Value |
|--------|-------|
| Training Time | 2-3 hours (V100 GPU) |
| Inference Time | 8-12 ms per window pair |
| Model Size | 2.6 MB (float32) |
| Memory Footprint | ~15 MB during inference |

---

## 9. Extensions for Our Research

### 9.1 What to Build On

1. **Siamese architecture concept** - Use for our self-supervised learning
2. **Feature combination strategy** - Adopt concatenation + element-wise ops
3. **Binary classification formulation** - Adapt for self-supervised pretext task

### 9.2 What to Improve

| Aspect | Current (DSS) | Proposed (SAS-HAR) | Improvement |
|--------|---------------|---------------------|-------------|
| Features | CNN only | CNN + Transformer | Long-range context |
| Learning | Supervised | Self-supervised | Label efficiency |
| Parameters | 650K | <25K (distilled) | 26x smaller |
| Edge Ready | No | Yes (quantized) | Deployable |

### 9.3 Key Gaps to Address

1. **Self-supervised learning:** Remove need for labeled boundaries
2. **Attention mechanisms:** Model long-range dependencies
3. **Edge optimization:** Distillation + quantization
4. **Transitional specialization:** Dedicated module for transitions

---

## 10. Key Equations Reference

### CNN Encoder
```
F = GAP(Conv3(BN(ReLU(Conv2(BN(ReLU(Conv1(W))))))))
```

### Similarity Computation
```
P = σ(MLP([Fᵢ; Fᵢ₊₁; Fᵢ⊙Fᵢ₊₁; |Fᵢ-Fᵢ₊₁|]))
```

### Loss Function
```
L = BCE(P, Y) = -1/N Σ [y log(p) + (1-y) log(1-p)]
```

---

## 11. Citation

```bibtex
@article{baraka2024deep,
  title={Deep Similarity Segmentation Model for Sensor-Based Activity Recognition},
  author={Baraka, AbdulRahman and Mohd Noor, Mohd Halim},
  journal={Multimedia Tools and Applications},
  volume={84},
  number={11},
  pages={8869--8892},
  year={2024},
  publisher={Springer},
  doi={10.1007/s11042-024-18933-2}
}
```

---

*Last Updated: March 2026*
