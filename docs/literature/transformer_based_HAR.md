# Transformer-Based Methods for HAR

## Overview

Transformers have emerged as a powerful architecture for Human Activity Recognition, offering advantages in modeling long-range temporal dependencies and parallel processing. This document surveys transformer-based approaches in HAR.

## Transformer Architecture Fundamentals

### Core Components

```
┌─────────────────────────────────────────┐
│            Transformer Encoder          │
├─────────────────────────────────────────┤
│  ┌─────────────────────────────────┐   │
│  │     Multi-Head Self-Attention   │   │
│  │                                 │   │
│  │  Q ──┐                          │   │
│  │  K ──┼──→ Attention ──→ Output  │   │
│  │  V ──┘                          │   │
│  └─────────────────────────────────┘   │
│              ↓ + Residual               │
│         Layer Normalization             │
│              ↓                          │
│  ┌─────────────────────────────────┐   │
│  │     Feed-Forward Network        │   │
│  │     (MLP with GELU)             │   │
│  └─────────────────────────────────┘   │
│              ↓ + Residual               │
│         Layer Normalization             │
└─────────────────────────────────────────┘
```

### Self-Attention Mechanism

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

**Key Properties:**
- $O(n^2)$ complexity in sequence length
- Captures global dependencies
- Interpretable attention weights

### Multi-Head Attention

$$\text{MultiHead}(Q, K, V) = \text{Concat}(head_1, ..., head_h)W^O$$

where $head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$

**Benefits:**
- Different heads capture different relationships
- Parallel computation of attention patterns
- Richer representations

## Adapting Transformers for Sensor Data

### Challenge: Positional Encoding for Time Series

Unlike language, sensor data has continuous time properties.

**Solutions:**

1. **Sinusoidal Encoding (Standard)**
   $$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d})$$
   $$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d})$$

2. **Learnable Positional Embedding**
   ```python
   self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model))
   ```

3. **Continuous Time Encoding**
   $$PE(t, 2i) = \sin(\omega_i \cdot t)$$
   $$PE(t, 2i+1) = \cos(\omega_i \cdot t)$$

### Challenge: Variable-Length Sequences

Sensor recordings have varying lengths.

**Solutions:**
- Padding with attention masks
- Sliding window approach
- Hierarchical processing

### Challenge: Multi-Channel Sensor Data

Multiple sensor axes require special handling.

**Approaches:**

1. **Channel Concatenation**
   ```python
   # Input: (batch, channels, time)
   x = x.permute(0, 2, 1)  # (batch, time, channels)
   x = self.linear(x)      # Project to d_model
   ```

2. **Separate Channel Embeddings**
   ```python
   # Process each channel separately, then combine
   channel_embeds = [self.embeds[ch](x[:, ch, :]) for ch in range(C)]
   x = torch.stack(channel_embeds, dim=1).mean(dim=1)
   ```

## Key Transformer Models for HAR

### 1. HAR-Transformer (2021)

**Architecture:**
```
Sensor Input (C channels × T timesteps)
         │
         ▼
┌─────────────────┐
│ Patch Embedding │  Divide into patches
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ + Positional    │
│   Encoding      │
└────────┬────────┘
         │
    ┌────▼────┐
    │ Encoder │ × L
    │ Layers  │
    └────┬────┘
         │
    ┌────▼────┐
    │  CLS    │
    │ Token   │
    └────┬────┘
         │
    ┌────▼────┐
    │  MLP    │
    │ Head    │
    └─────────┘
```

**Key Results:**
- UCI-HAR: 96.5%
- WISDM: 91.2%

### 2. ActivityFormer (2023)

**Innovation:** Temporal action localization for HAR

**Architecture:**
- Feature pyramid for multi-scale processing
- Deformable attention for efficient computation
- Classification + regression heads

**Performance:**
- End-to-end localization
- Handles variable-length activities

### 3. Vita-Transformer (2024)

**Innovation:** Vision-transformer inspired for IMU data

**Key Features:**
- Patch-based sensor processing
- Cross-attention between modalities
- Lightweight design (0.5M parameters)

## Efficient Transformer Variants

### Linear Attention Transformers

Reduce $O(n^2)$ to $O(n)$ complexity:

$$\text{LinearAttention}(Q, K, V) = \phi(Q)(\phi(K)^T V)$$

where $\phi$ is a kernel function (e.g., elu + 1).

### Sparse Attention

Attend only to relevant positions:

```
Full Attention:     All positions attend to all
Sparse Attention:   Each position attends to subset

Patterns:
- Local: Nearby positions
- Strided: Every k positions
- Random: Random subset
```

### Performer (2024)

Uses random features for kernel approximation:

$$\text{Attention}(Q, K, V) \approx \text{softmax}'(Q) \cdot (\text{softmax}'(K)^T V)$$

**Benefits:**
- Linear complexity
- No approximation error bound needed
- Works well for HAR sequences

## Self-Attention for Segmentation

### Temporal Segmentation with Transformers

```
Input Sequence:  [----Activity A----][----Activity B----]
                  ↑                   ↑
Attention Peaks:  Low                 High (transition)
```

### Boundary Detection via Attention

```python
class BoundaryTransformer(nn.Module):
    def __init__(self, d_model, n_heads):
        self.transformer = TransformerEncoder(d_model, n_heads)
        self.boundary_head = nn.Linear(d_model, 1)
    
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        attended = self.transformer(x)
        boundary_scores = torch.sigmoid(self.boundary_head(attended))
        return boundary_scores
```

### Cross-Attention for Change Detection

Comparing adjacent windows:

$$\text{CrossAttn}(W_t, W_{t+1}) = \text{Attention}(Q_t, K_{t+1}, V_{t+1})$$

High attention divergence indicates transition.

## Comparison with CNN/RNN Approaches

| Aspect | CNN | RNN/LSTM | Transformer |
|--------|-----|----------|-------------|
| **Receptive Field** | Local (limited) | Sequential (slow) | Global (direct) |
| **Parallelization** | High | Low | High |
| **Long Dependencies** | Poor | Moderate | Excellent |
| **Parameter Efficiency** | High | Moderate | Lower |
| **Training Stability** | Good | Variable | Good |
| **Interpretability** | Low | Low | High (attention) |

## State-of-the-Art Transformer HAR (2024-2025)

### P2LHAP (2024)

**Innovation:** Prompt-to-learn for activity segmentation

**Architecture:**
- Learnable prompts for different activity types
- Boundary-aware attention mechanism
- Handles transitional activities

**Results:**
- Boundary F1: 95.7%
- First to exceed 95% on standard benchmarks

### Temporal Fusion Transformer

**Innovation:** Combining attention mechanisms

```
Variable Selection → Self-Attention → Gated Residual
       ↑                                    │
       └────────────────────────────────────┘
```

**Benefits:**
- Static and dynamic feature handling
- Interpretable feature importance
- Uncertainty quantification

### Efficient Transformers for Edge

| Model | Params | Latency | Accuracy |
|-------|--------|---------|----------|
| MobileBERT-HAR | 1.2M | 8ms | 94.8% |
| EdgeFormer | 0.8M | 5ms | 93.5% |
| Tiny-ViT-HAR | 0.5M | 3ms | 92.1% |

## Self-Supervised Transformers for HAR

### Masked Sensor Modeling

```
Original:  [A, B, C, D, E, F, G, H]
Masked:    [A, _, C, _, E, F, _, H]
Task:      Predict masked positions
```

### Contrastive Learning

```python
def contrastive_loss(z1, z2, temperature=0.1):
    # z1, z2: augmented views of same sample
    similarity = F.cosine_similarity(z1.unsqueeze(1), 
                                      z2.unsqueeze(0), dim=-1)
    labels = torch.arange(z1.size(0))
    return F.cross_entropy(similarity / temperature, labels)
```

### Pre-training → Fine-tuning Paradigm

```
┌────────────────────────────────────┐
│   Self-Supervised Pre-training     │
│   (Large unlabeled sensor data)    │
└─────────────────┬──────────────────┘
                  │
                  ▼
┌────────────────────────────────────┐
│   Supervised Fine-tuning           │
│   (Limited labeled activity data)  │
└────────────────────────────────────┘
```

## Challenges and Solutions

### 1. Computational Cost

**Challenge:** $O(n^2)$ attention complexity

**Solutions:**
- Linear attention variants
- Sparse attention patterns
- Hierarchical processing

### 2. Data Requirements

**Challenge:** Transformers need large datasets

**Solutions:**
- Self-supervised pre-training
- Data augmentation strategies
- Transfer learning from related domains

### 3. Real-Time Constraints

**Challenge:** Edge deployment limitations

**Solutions:**
- Model distillation
- Quantization-aware training
- Efficient architectures (MobileBERT style)

## Relation to Our Research

### SAS-HAR Transformer Design

1. **Hybrid CNN-Transformer**
   ```
   CNN Encoder → Transformer Encoder → Task Heads
   (Local)        (Global)            (Segmentation)
   ```

2. **Boundary-Aware Attention**
   - Learn to attend to transition points
   - Self-supervised boundary discovery

3. **Efficient Design**
   - Linear attention for real-time processing
   - Knowledge distillation for edge deployment

### Novel Contributions

| Aspect | Standard Transformers | SAS-HAR |
|--------|----------------------|---------|
| **Segmentation** | Post-hoc | Integrated |
| **Training** | Supervised | Self-supervised (TCBL) |
| **Attention** | General | Boundary-focused |
| **Efficiency** | Computation-heavy | Edge-optimized |

## Code Example: HAR Transformer

```python
class HARTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Input embedding
        self.patch_embed = nn.Linear(config.channels, config.d_model)
        self.pos_embed = nn.Parameter(
            torch.randn(1, config.max_seq_len, config.d_model)
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=config.n_layers
        )
        
        # Task heads
        self.activity_head = nn.Linear(config.d_model, config.n_classes)
        self.boundary_head = nn.Linear(config.d_model, 1)
    
    def forward(self, x):
        # x: (batch, channels, time)
        B, C, T = x.shape
        
        # Embed
        x = x.permute(0, 2, 1)  # (B, T, C)
        x = self.patch_embed(x)  # (B, T, d_model)
        x = x + self.pos_embed[:, :T, :]
        
        # Transform
        x = self.transformer(x)
        
        # Predict
        activity = self.activity_head(x)
        boundary = torch.sigmoid(self.boundary_head(x))
        
        return activity, boundary
```

## References

1. Vaswani, A., et al. (2017). "Attention Is All You Need." *NeurIPS*.
2. Dosovitskiy, A., et al. (2021). "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." *ICLR*.
3. Zeng, H., et al. (2022). "HAR-Transformer: Hierarchical Activity Recognition Using Transformer." *Sensors*.
4. Li, Y., et al. (2023). "ActivityFormer: Temporal Action Localization for Wearable Sensors." *AAAI*.
5. Zhang, H., et al. (2024). "P2LHAP: Prompt-to-Learn Human Activity Prediction." *CVPR*.
6. Wang, S., et al. (2024). "Efficient Transformers for Time Series Classification." *ICML*.

## Future Directions

1. **Pre-trained Foundation Models:** Large-scale pre-training for HAR
2. **Multimodal Fusion:** Combining video, audio, and sensors
3. **Adaptive Computation:** Dynamic computation based on complexity
4. **Interpretable Predictions:** Attention-based explanations
5. **Federated Learning:** Privacy-preserving distributed training
