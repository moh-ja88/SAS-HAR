# Literature Review: Self-Supervised Learning for HAR

## Overview

Self-supervised learning (SSL) enables learning meaningful representations from unlabeled data by solving pretext tasks. This document reviews SSL methods applicable to HAR and temporal segmentation.

---

## 1. Self-Supervised Learning Paradigm

### 1.1 Core Concept

```
Traditional Supervised Learning:
  Labeled Data → Model → Predictions

Self-Supervised Learning:
  Unlabeled Data → Pretext Task → Learned Representations
                              ↓
                   Fine-tune with Few Labels
```

### 1.2 Advantages for HAR

| Advantage | Impact |
|-----------|--------|
| **Reduced annotation cost** | 90%+ label reduction |
| **Better generalization** | Transfer across datasets |
| **Scalability** | Leverage large unlabeled data |
| **Privacy** | Local pre-training possible |

---

## 2. Contrastive Learning Methods

### 2.1 SimCLR Framework

**Principle:** Learn representations by contrasting augmented views.

```
1. Sample batch of windows
2. Create two augmented views per window
3. Contrast positive pairs vs. negative pairs
4. Learn to bring positive pairs together
```

**Augmentations for Sensors:**
- Time warping
- Jittering (Gaussian noise)
- Scaling
- Rotation (for 3D sensors)
- Channel dropout

**Loss Function:**
```
L = -log(exp(sim(zᵢ, zⱼ)/τ) / Σₖ exp(sim(zᵢ, zₖ)/τ))
```

### 2.2 MoCo (Momentum Contrast)

**Principle:** Use momentum-updated encoder and large queue.

```
1. Query encoder: f_q (trained with gradient)
2. Key encoder: f_k (momentum update: θ_k = m·θ_k + (1-m)·θ_q)
3. Queue: Store past keys as negatives
4. Contrast queries against keys
```

**Advantage:** Large number of negatives without large batch size.

### 2.3 BYOL (Bootstrap Your Own Latent)

**Principle:** No negative pairs needed - self-distillation.

```
1. Online network: encoder + projector + predictor
2. Target network: encoder + projector (momentum update)
3. Predict target representation from online
4. No explicit negatives
```

---

## 3. Masked Modeling Methods

### 3.1 Masked Language Modeling (BERT-style)

**Principle:** Mask random tokens, predict from context.

**Adaptation for Sensors:**
```
1. Mask 15% of timesteps
2. Replace with [MASK] token or zeros
3. Predict original values
4. Loss: MSE on masked positions
```

### 3.2 Masked Autoencoders (MAE)

**Principle:** High masking ratio (75%), asymmetric encoder-decoder.

```
Encoder:
  - Process only visible patches
  - Lightweight

Decoder:
  - Process visible + mask tokens
  - Reconstruct original
```

**For HAR:** Mask time steps, predict sensor values.

---

## 4. Temporal Contrastive Learning

### 4.1 Temporal Contrastive Learning (TCL)

**Principle:** Contrast segments from different times.

```
Positive pairs: Segments close in time
Negative pairs: Segments far apart or from different users
```

**Key Insight:** Temporal proximity implies semantic similarity.

### 4.2 Time-Contrastive Learning (TCL)

**Principle:** Use time as supervision signal.

```
1. Divide timeline into bins
2. Classify which bin a segment belongs to
3. Learn temporal features
```

---

## 5. SSL for HAR: Existing Work

### 5.1 LIMU-BERT

**Pre-training:** Masked sensor modeling
**Fine-tuning:** Activity classification
**Results:** 95% accuracy with 10% labels

### 5.2 SelfHAR

**Pre-training:** Contrastive learning + teacher-student
**Results:** Good transfer across devices

### 5.3 Context-Aware SSL

**Pre-training:** Predict context (location, time)
**Results:** Better generalization

---

## 6. Proposed: Temporal Contrastive Boundary Learning (TCBL)

### 6.1 Novel Pretext Tasks

#### Task 1: Temporal Contrastive Learning

```
Objective: Cluster same-activity segments

Positive pairs: Adjacent segments (likely same activity)
Negative pairs: Distant segments (likely different activities)

Loss: InfoNCE with temporal weighting
```

#### Task 2: Continuity Prediction

```
Objective: Predict if segments are continuous

Input: Pair of adjacent segments
Output: Binary (same activity or boundary)

Training signal: Pseudo-labels from signal variance
```

#### Task 3: Masked Temporal Modeling

```
Objective: Predict masked timesteps

Input: Partially masked segment
Output: Reconstructed segment

Loss: MSE on masked positions
```

### 6.2 Combined Loss

```
L_TCBL = λ₁·L_contrastive + λ₂·L_continuity + λ₃·L_masked

Default: λ₁=1.0, λ₂=0.5, λ₃=0.3
```

### 6.3 Expected Results

| Labels | Random Init | TCBL Pre-trained | Improvement |
|--------|-------------|------------------|-------------|
| 1% | 55% | 78% | +23% |
| 5% | 68% | 86% | +18% |
| 10% | 75% | 90% | +15% |
| 100% | 95% | 98% | +3% |

---

## 7. Comparison with Existing SSL

| Method | Task | Pre-text | HAR-specific? | Segmentation? |
|--------|------|----------|---------------|---------------|
| SimCLR | Classification | Contrast | No | No |
| MoCo | Classification | Contrast | No | No |
| BERT | NLP | Masked | No | No |
| LIMU-BERT | HAR | Masked | Yes | No |
| SelfHAR | HAR | Multi-task | Yes | No |
| **TCBL (Ours)** | **Segmentation** | **Temporal** | **Yes** | **Yes** |

---

## 8. Implementation Considerations

### 8.1 Augmentation Strategy

```python
def augment_sensor_data(window):
    augmentations = [
        time_warp(window, factor=random(0.8, 1.2)),
        jitter(window, sigma=0.01),
        scale(window, factor=random(0.9, 1.1)),
        rotate(window, angle=random(-15, 15)),  # For 3D
        dropout(window, rate=0.1)  # Channel dropout
    ]
    return random.choice(augmentations)
```

### 8.2 Batch Construction

```python
def create_ssl_batch(stream, window_size, batch_size):
    batch = []
    for _ in range(batch_size):
        # Sample segment
        start = random.randint(0, len(stream) - window_size)
        segment = stream[start:start+window_size]
        
        # Create positive pair (adjacent)
        adjacent = stream[start+window_size//2:start+3*window_size//2]
        
        batch.append((segment, adjacent))
    
    return batch
```

### 8.3 Training Protocol

```
Phase 1: Self-supervised pre-training
  - 100 epochs on unlabeled data
  - Learning rate: 1e-4
  - Batch size: 64

Phase 2: Supervised fine-tuning
  - 50 epochs with task labels
  - Learning rate: 1e-5
  - Labels: 1%, 5%, 10%, 50%, 100%
```

---

## 9. Research Gaps

### 9.1 Current Limitations

1. **No SSL for segmentation** - All methods focus on classification
2. **Boundary pretext tasks** - Not explored
3. **Temporal continuity** - Not explicitly modeled
4. **Transfer learning** - Limited across HAR domains

### 9.2 Opportunities

1. **Boundary-aware SSL** - Novel contribution
2. **Multi-task SSL** - Combine multiple pretext tasks
3. **Hierarchical SSL** - Multi-scale representations
4. **Cross-modal SSL** - Learn from multiple sensors

---

## References

1. Chen et al. (2020). SimCLR: Simple framework for contrastive learning.
2. He et al. (2020). MoCo: Momentum contrast.
3. Grill et al. (2020). BYOL: Bootstrap your own latent.
4. Devlin et al. (2019). BERT: Pre-training of deep bidirectional transformers.
5. He et al. (2022). MAE: Masked autoencoders.

---

*Last Updated: March 2026*
