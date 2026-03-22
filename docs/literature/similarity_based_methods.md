# Similarity-Based Methods for Activity Segmentation

## Overview

Similarity-based segmentation methods represent a fundamental approach to temporal segmentation in Human Activity Recognition (HAR). These methods leverage the principle that similar sensor patterns belong to the same activity class, enabling boundary detection through similarity analysis.

## Core Principles

### 1. Similarity Metrics

Similarity-based methods rely on distance or similarity measures to compare sensor windows:

| Metric | Formula | Characteristics |
|--------|---------|-----------------|
| Euclidean Distance | $\sqrt{\sum_{i}(x_i - y_i)^2}$ | Simple, sensitive to magnitude |
| Cosine Similarity | $\frac{x \cdot y}{\|x\|\|y\|}$ | Scale-invariant, direction-focused |
| Dynamic Time Warping (DTW) | $\min_{path} \sum d(x_i, y_j)$ | Time-shift invariant |
| Pearson Correlation | $\frac{\sum(x_i-\bar{x})(y_i-\bar{y})}{\sigma_x \sigma_y}$ | Linear relationship measure |

### 2. Window-Based Comparison

The general approach involves:

1. **Sliding Window**: Extract overlapping windows from sensor streams
2. **Feature Extraction**: Compute window representations
3. **Similarity Computation**: Compare consecutive windows
4. **Boundary Detection**: Identify transitions where similarity drops below threshold

```
Window n    Window n+1   Similarity
[---A---]   [---B---]   → Low (Boundary)
[---A---]   [---A---]   → High (Same Activity)
```

## Key Methods in Literature

### 1. Similarity Segmentation (Baraka et al., 2023)

**Core Algorithm:**
```
For each consecutive window pair (W_i, W_{i+1}):
    1. Extract statistical features
    2. Compute similarity score S_i
    3. If S_i < threshold τ:
         Mark boundary between W_i and W_{i+1}
```

**Features Used:**
- Mean, variance, skewness, kurtosis
- Spectral features (FFT coefficients)
- Zero-crossing rate
- Signal magnitude area

**Strengths:**
- Interpretable boundaries
- No training required
- Works on unseen activities

**Limitations:**
- Fixed threshold sensitivity
- Manual feature engineering
- Limited temporal context

### 2. Deep Similarity Segmentation (Baraka et al., 2024)

**Architecture:**
```
Input Windows → CNN Encoder → Learned Embeddings → Similarity Network → Boundary Score
```

**Key Innovations:**
1. **Learned Representations**: CNN replaces hand-crafted features
2. **Trainable Similarity**: Neural network learns optimal similarity function
3. **Multi-scale Analysis**: Hierarchical feature extraction

**Loss Function:**
$$\mathcal{L} = \text{BCE}(y_{boundary}, \hat{y}_{boundary}) + \lambda \cdot \text{ContrastiveLoss}$$

**Performance:**
- 15-20% improvement over traditional methods
- Better generalization to new activities
- Handles gradual transitions better

### 3. Statistical Change Detection

**CUSUM (Cumulative Sum):**
$$C_n = \max(0, C_{n-1} + x_n - \mu_0 - k)$$

- Detects mean shifts in sensor signals
- Low computational complexity
- Sensitive to threshold selection

**Likelihood Ratio Test:**
$$\Lambda = \frac{p(x | H_1)}{p(x | H_0)}$$

- Hypothesis testing framework
- Requires distribution assumptions
- Theoretical optimality guarantees

## Similarity vs. Classification-Based Segmentation

| Aspect | Similarity-Based | Classification-Based |
|--------|------------------|---------------------|
| **Training** | Unsupervised/Minimal | Supervised Required |
| **Novel Activities** | Better handling | Requires retraining |
| **Boundary Precision** | Moderate | Higher (with training) |
| **Computational Cost** | Lower | Higher |
| **Interpretability** | High | Lower |

## Challenges in Similarity-Based Methods

### 1. Threshold Selection

```python
# Challenge: Finding optimal threshold
threshold = ?  # Dataset-dependent

# Solutions:
# - Adaptive thresholds based on local statistics
# - Learning thresholds from validation data
# - Multiple thresholds for different activity types
```

### 2. Gradual Transitions

Transitional activities (walking→running) create ambiguous similarity patterns:

```
Activity A → Transition → Activity B
Similarity: High → Medium → Low
```

### 3. Within-Activity Variability

Same activity performed differently:
- Fast walking vs. slow walking
- Different users
- Sensor placement variations

### 4. Temporal Scale Selection

Window size affects segmentation quality:
- Too small: Noise sensitivity
- Too large: Boundary precision loss

## Recent Advances (2023-2025)

### 1. Adaptive Similarity Thresholds

Dynamic thresholds based on:
- Local activity statistics
- Sliding window variance
- Learned uncertainty estimates

### 2. Multi-Modal Similarity

Combining similarity scores from multiple sensors:
$$S_{combined} = \sum_{m} w_m \cdot S_m$$

Where $m$ indexes modalities (accelerometer, gyroscope, etc.)

### 3. Hierarchical Similarity

Multi-resolution similarity analysis:
- Fine-grained: Detect rapid transitions
- Coarse-grained: Identify activity groups

### 4. Attention-Enhanced Similarity

Using attention mechanisms to weight relevant features:
$$S_{att} = \text{softmax}(QK^T) V$$

## Relation to Our Research

### Building on Baraka's Work

Our SAS-HAR framework extends similarity-based methods by:

1. **Attention-Based Similarity**
   - Replace fixed similarity with learnable attention
   - Dynamic feature weighting per window

2. **Self-Supervised Learning**
   - TCBL learns boundary-aware representations
   - No labeled boundaries required

3. **Temporal Consistency**
   - Long-range dependencies via Transformer
   - Beyond local window comparison

### Key Differentiators

| Aspect | Baraka 2024 | SAS-HAR (Ours) |
|--------|-------------|----------------|
| Similarity Function | Fixed architecture | Attention-based |
| Training | Supervised boundaries | Self-supervised |
| Context | Local windows | Global temporal |
| Threshold | Manual/Validation | Learned |

## References

1. Baraka, A., et al. (2023). "Similarity Segmentation Approach for Sensor-Based Activity Recognition." *Sensors*.
2. Baraka, A., et al. (2024). "Deep Similarity Segmentation Model for Sensor-Based Activity Recognition." *IEEE Sensors Journal*.
3. Noor, M.H.M., et al. (2017). "Adaptive Sliding Window Segmentation for Physical Activity Recognition." *Sensors*.
4. Basseville, M., & Nikiforov, I.V. (1993). "Detection of Abrupt Changes: Theory and Application."
5. Aminikhanghahi, S., & Cook, D. J. (2017). "A Survey of Methods for Time Series Change Point Detection." *Knowledge and Information Systems*.

## Future Directions

1. **Cross-User Similarity**: Personalized similarity metrics
2. **Transfer Learning**: Similarity functions across domains
3. **Online Adaptation**: Real-time threshold updates
4. **Multi-Task Learning**: Joint segmentation and recognition
