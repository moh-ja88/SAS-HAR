# Deep Learning for Human Activity Recognition

## Overview

Deep learning has revolutionized Human Activity Recognition (HAR) by enabling automatic feature learning from raw sensor data. This document surveys the evolution and state-of-the-art in deep learning approaches for HAR.

## Evolution of HAR Methods

```
Traditional ML (2000-2012)     Deep Learning Era (2012-Present)
┌─────────────────────┐       ┌─────────────────────────────────┐
│ Manual Features     │       │ Learned Features                │
│ • Statistical       │  →    │ • CNN: Spatial patterns         │
│ • Spectral          │       │ • RNN: Temporal dynamics        │
│ • Time-domain       │       │ • Transformer: Long-range deps  │
│                     │       │                                 │
│ Classifiers:        │       │ End-to-End Learning             │
│ SVM, RF, HMM        │       │ Raw Signal → Activity Label     │
└─────────────────────┘       └─────────────────────────────────┘
```

## Convolutional Neural Networks (CNNs)

### 1D-CNN Architecture

```python
class HAR_CNN(nn.Module):
    def __init__(self):
        # Input: (batch, channels, time_steps)
        self.conv1 = nn.Conv1d(3, 64, kernel_size=5)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3)
        self.pool = nn.MaxPool1d(2)
        self.fc = nn.Linear(256 * reduced_time, num_classes)
```

### Key CNN Approaches

| Method | Architecture | Key Innovation | Accuracy (UCI-HAR) |
|--------|--------------|----------------|-------------------|
| DeepConvLSTM | CNN + LSTM | Hybrid architecture | 95.3% |
| Inception-HAR | Multi-scale CNN | Parallel convolutions | 96.1% |
| ResNet-HAR | Residual connections | Deep networks | 95.8% |
| TinyHAR | Lightweight CNN | Edge deployment | 93.2% |

### Multi-Scale CNNs

Capturing patterns at different temporal scales:

```
                    ┌── Conv 3x1 ──┐
Input ──┬── Conv 5x1 ──┼──────────────┼── Concat ── Output
        └── Conv 7x1 ──┴──────────────┘

Benefits:
- Short activities (falling): Small kernels
- Long activities (walking): Large kernels
```

### Dilated Convolutions

Expanding receptive field without parameter increase:

$$\text{Receptive Field} = 1 + \sum_{i=1}^{L}(k_i - 1) \cdot d_i$$

Where $d_i$ is dilation rate at layer $i$.

## Recurrent Neural Networks (RNNs)

### LSTM for HAR

```python
class HAR_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
    
    def forward(self, x):
        # x: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])
```

### LSTM Gate Mechanism

$$
\begin{aligned}
f_t &= \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) & \text{(Forget gate)} \\
i_t &= \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) & \text{(Input gate)} \\
o_t &= \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) & \text{(Output gate)} \\
\tilde{C}_t &= \tanh(W_C \cdot [h_{t-1}, x_t] + b_C) & \text{(Candidate)} \\
C_t &= f_t \odot C_{t-1} + i_t \odot \tilde{C}_t & \text{(Cell state)} \\
h_t &= o_t \odot \tanh(C_t) & \text{(Hidden state)}
\end{aligned}
$$

### Bidirectional RNNs

Processing sequences in both directions:

```
Forward:  t1 → t2 → t3 → t4 → t5
Backward: t1 ← t2 ← t3 ← t4 ← t5
Combined: [h_forward; h_backward]
```

**Advantage**: Future context informs past predictions

## Hybrid Architectures

### CNN-LSTM (DeepConvLSTM)

```
Raw Signal
    │
    ▼
┌─────────┐
│ CNN     │ ← Local feature extraction
│ Layers  │
└────┬────┘
     │
     ▼
┌─────────┐
│ LSTM    │ ← Temporal modeling
│ Layers  │
└────┬────┘
     │
     ▼
┌─────────┐
│ Dense   │ ← Classification
│ Layer   │
└─────────┘
```

### Performance Comparison

| Dataset | CNN Only | LSTM Only | CNN-LSTM |
|---------|----------|-----------|----------|
| UCI-HAR | 94.2% | 91.5% | 95.8% |
| WISDM | 88.3% | 86.1% | 90.2% |
| PAMAP2 | 91.4% | 89.7% | 93.1% |

## Attention Mechanisms

### Self-Attention for HAR

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

### Temporal Attention

Weighting time steps by importance:

```python
class TemporalAttention(nn.Module):
    def __init__(self, hidden_size):
        self.attention = nn.Linear(hidden_size, 1)
    
    def forward(self, lstm_output):
        # lstm_output: (batch, seq_len, hidden)
        weights = F.softmax(self.attention(lstm_output), dim=1)
        context = (lstm_output * weights).sum(dim=1)
        return context, weights
```

### Multi-Head Attention

Parallel attention for different aspects:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(head_1, ..., head_h) W^O$$

Where $head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$

## Transformer-Based HAR

### Architecture Overview

```
Input Embedding
       │
       ▼
┌──────────────────┐
│ Positional       │
│ Encoding         │
└────────┬─────────┘
         │
    ┌────▼────┐
    │ Encoder │ × N layers
    │ Block   │
    └────┬────┘
         │
    ┌────▼────┐
    │ Pooling │
    └────┬────┘
         │
    ┌────▼────┐
    │   MLP   │
    └─────────┘
```

### Key Transformer Models for HAR

| Model | Parameters | Key Feature | Accuracy |
|-------|------------|-------------|----------|
| HAR-Transformer | 1.2M | Position embedding | 96.5% |
| ViT-HAR | 850K | Patch-based | 95.9% |
| ActionFormer | 2.1M | Temporal localization | 97.2% |

### Positional Encoding

$$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}})$$
$$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}})$$

## State-of-the-Art Models (2024-2025)

### 1. TinierHAR (2024)

- **Parameters**: 34K (smallest)
- **Architecture**: Depthwise separable convolutions
- **Optimization**: Neural architecture search
- **Accuracy**: 93.2% on UCI-HAR

### 2. P2LHAP (2024)

- **Focus**: Activity segmentation
- **Innovation**: Prompt-to-learn framework
- **Boundary F1**: 95.7%
- **Key**: Handles transitional activities

### 3. nanoML-HAR (2025)

- **Energy**: 56-104 nJ/sample
- **Deployment**: Microcontroller optimized
- **Latency**: <5ms inference

## Comparison of Deep Learning Paradigms

| Paradigm | Strengths | Weaknesses | Best For |
|----------|-----------|------------|----------|
| **CNN** | Local patterns, parallelizable | Limited temporal context | Static activities |
| **RNN/LSTM** | Sequential modeling | Slow training, vanishing gradients | Temporal sequences |
| **Transformer** | Long-range dependencies | Data hungry, compute intensive | Complex activities |
| **Hybrid** | Combines strengths | Complex architecture | General-purpose |

## Training Strategies

### 1. Data Augmentation

```python
augmentations = {
    'jitter': add_gaussian_noise,
    'scaling': random_scale,
    'rotation': rotate_sensor_axes,
    'time_warp': warp_time_axis,
    'magnitude_warp': warp_signal_magnitude
}
```

### 2. Transfer Learning

```
Pre-training (Large Dataset)
        │
        ▼
Fine-tuning (Target Dataset)
```

### 3. Multi-Task Learning

```python
class MultiTaskHAR(nn.Module):
    def __init__(self):
        self.shared_encoder = CNNEncoder()
        self.activity_head = nn.Linear(hidden, num_activities)
        self.user_head = nn.Linear(hidden, num_users)
        self.position_head = nn.Linear(hidden, num_positions)
```

## Challenges in Deep Learning HAR

### 1. Data Scarcity

- Labeled HAR data is expensive
- Solution: Self-supervised pre-training

### 2. Domain Shift

- User variations
- Sensor placement differences
- Solution: Domain adaptation

### 3. Class Imbalance

- Common activities dominate
- Solution: Focal loss, oversampling

### 4. Computational Constraints

- Edge deployment requirements
- Solution: Model compression, quantization

## Relation to Our Research

### SAS-HAR Architecture Choices

1. **Hybrid CNN-Transformer**
   - CNN: Local sensor pattern extraction
   - Transformer: Long-range temporal dependencies

2. **Attention-Based Design**
   - Boundary attention for segmentation
   - Self-attention for feature weighting

3. **Self-Supervised Pre-Training**
   - TCBL for label-efficient learning
   - Contrastive learning on unlabeled data

### Novel Contributions

| Aspect | Existing Methods | SAS-HAR |
|--------|------------------|---------|
| Segmentation | Post-hoc or sliding window | Integrated attention-based |
| Supervision | Full labels required | Self-supervised pre-training |
| Temporal Context | Limited (CNN) or slow (RNN) | Efficient (Transformer) |
| Edge Deployment | Often overlooked | Built-in optimization |

## References

1. Ordóñez, F. J., & Roggen, D. (2016). "Deep Convolutional and LSTM Recurrent Neural Networks for Multimodal Wearable Activity Recognition." *Sensors*.
2. Ronao, C. A., & Cho, S. B. (2016). "Human Activity Recognition with Smartphone Sensors Using Deep Learning Neural Networks." *Expert Systems with Applications*.
3. Wang, J., et al. (2019). "Deep Learning for Sensor-based Activity Recognition: A Survey." *Pattern Recognition Letters*.
4. Yao, S., et al. (2018). "DeepSense: A Unified Deep Learning Framework for Time-Series Mobile Sensing Data Processing." *WWW*.
5. Liu, H., et al. (2024). "TinierHAR: A Lightweight Deep Learning Architecture for Human Activity Recognition." *IEEE IoT Journal*.
