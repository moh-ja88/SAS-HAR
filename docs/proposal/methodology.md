# Technical Methodology

## SAS-HAR Framework Architecture

---

## 1. System Overview

The Self-Supervised Attention-based Segmentation for HAR (SAS-HAR) framework consists of five integrated modules:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SAS-HAR Framework                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Input: Continuous Sensor Stream [B, C, T]                          │
│         B = batch size, C = channels (6), T = timesteps             │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │  Module 1: CNN Feature Encoder                              │     │
│  │  • Local spatial feature extraction                         │     │
│  │  • Depthwise separable convolutions                         │     │
│  │  • Output: [B, 256, T']                                     │     │
│  └────────────────────────────────────────────────────────────┘     │
│                           │                                          │
│                           ▼                                          │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │  Module 2: Transformer Temporal Module                      │     │
│  │  • Long-range temporal dependencies                         │     │
│  │  • Efficient linear attention (O(n))                        │     │
│  │  • Output: [B, T', 256]                                     │     │
│  └────────────────────────────────────────────────────────────┘     │
│                           │                                          │
│              ┌────────────┴────────────┐                            │
│              ▼                         ▼                            │
│  ┌──────────────────────┐  ┌──────────────────────┐                │
│  │ Module 3: Boundary   │  │ Module 4: Transition │                │
│  │ Detection Head       │  │ Specialization       │                │
│  │ • Semantic attention │  │ • Multi-scale conv   │                │
│  │ • Boundary probs     │  │ • Dynamic attention  │                │
│  │ • Output: [B, T', 1] │  │ • Output: [B, T', 64]│                │
│  └──────────────────────┘  └──────────────────────┘                │
│              │                         │                            │
│              └────────────┬────────────┘                            │
│                           ▼                                          │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │  Module 5: Classification Head                              │     │
│  │  • Activity classification                                  │     │
│  │  • Output: [B, num_classes]                                 │     │
│  └────────────────────────────────────────────────────────────┘     │
│                           │                                          │
│                           ▼                                          │
│  Output: Boundaries + Activity Labels                               │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘

Training Modes:
┌─────────────────────────────────────────────────────────────────────┐
│  Mode 1: Self-Supervised Pre-training                               │
│  • Temporal Contrastive Boundary Learning (TCBL)                    │
│  • No labels required                                               │
│  • Pretext tasks: contrastive + continuity + masked                 │
├─────────────────────────────────────────────────────────────────────┤
│  Mode 2: Supervised Fine-tuning                                     │
│  • Activity labels + boundary labels                                │
│  • Joint segmentation + classification loss                         │
├─────────────────────────────────────────────────────────────────────┤
│  Mode 3: Knowledge Distillation                                     │
│  • Teacher-student training                                         │
│  • Target: <25K parameters                                          │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Module 1: CNN Feature Encoder

### Purpose
Extract local spatial features from raw sensor windows efficiently.

### Architecture Details

```python
class CNNFeatureEncoder(nn.Module):
    """
    Efficient CNN encoder using depthwise separable convolutions
    
    Input: [batch, 6, T]  # 6 channels: acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
    Output: [batch, 256, T']  # T' = T/8 due to pooling
    """
    
    def __init__(self, input_channels=6, hidden_dims=[64, 128, 256]):
        super().__init__()
        
        # Block 1: 6 -> 64
        self.block1 = nn.Sequential(
            DepthwiseSeparableConv1d(input_channels, hidden_dims[0], kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2, stride=2)  # T/2
        )
        
        # Block 2: 64 -> 128
        self.block2 = nn.Sequential(
            DepthwiseSeparableConv1d(hidden_dims[0], hidden_dims[1], kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2, stride=2)  # T/4
        )
        
        # Block 3: 128 -> 256
        self.block3 = nn.Sequential(
            DepthwiseSeparableConv1d(hidden_dims[1], hidden_dims[2], kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dims[2]),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2, stride=2)  # T/8
        )
        
    def forward(self, x):
        # x: [batch, 6, T]
        x = self.block1(x)  # [batch, 64, T/2]
        x = self.block2(x)  # [batch, 128, T/4]
        x = self.block3(x)  # [batch, 256, T/8]
        return x
```

### Depthwise Separable Convolution

```python
class DepthwiseSeparableConv1d(nn.Module):
    """
    Depthwise separable convolution for efficiency
    
    Standard Conv: params = C_in × C_out × K
    Depthwise Separable: params = C_in × K + C_in × C_out
    
    For K=3, C_in=64, C_out=128:
    Standard: 64 × 128 × 3 = 24,576
    Separable: 64 × 3 + 64 × 128 = 8,384 (3x reduction)
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super().__init__()
        
        # Depthwise: spatial convolution per channel
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, 
            kernel_size=kernel_size, 
            padding=padding, 
            groups=in_channels  # Key: separate per channel
        )
        
        # Pointwise: 1x1 convolution for channel mixing
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
```

### Parameter Count

| Layer | Parameters |
|-------|-----------|
| Block 1 | 1,216 |
| Block 2 | 12,544 |
| Block 3 | 49,664 |
| **Total** | **63,424** |

---

## 3. Module 2: Transformer Temporal Module

### Purpose
Model long-range temporal dependencies for boundary detection.

### Efficient Linear Attention

Standard attention has O(n²) complexity. We use linear attention with O(n) complexity:

```python
class EfficientLinearAttention(nn.Module):
    """
    Linear attention with O(n) complexity
    
    Standard: Attention(Q,K,V) = softmax(QK^T/√d)V  → O(n²)
    Linear: Attention(Q,K,V) = φ(Q)(φ(K)^T V) / φ(Q)φ(K)^T  → O(n)
    
    where φ is a kernel function (elu + 1)
    """
    
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, T, C = x.shape
        
        # Project to Q, K, V
        Q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply kernel function (elu + 1)
        Q = F.elu(Q) + 1
        K = F.elu(K) + 1
        
        # Linear attention: O(n) instead of O(n²)
        # Compute K^T V first: [B, H, D, D]
        KV = torch.einsum('bhnd,bhne->bhde', K, V)
        
        # Compute Q(K^T V): [B, H, N, D]
        Q_KV = torch.einsum('bhnd,bhde->bhne', Q, KV)
        
        # Compute normalizer: Q K^T 1
        K_sum = K.sum(dim=2, keepdim=True)  # [B, H, 1, D]
        normalizer = torch.einsum('bhnd,bhkd->bhnk', Q, K_sum).squeeze(-1)
        
        # Normalize
        out = Q_KV / (normalizer.unsqueeze(-1) + 1e-6)
        
        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(out)
        
        return self.dropout(out)
```

### Full Transformer Block

```python
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, mlp_ratio=2, dropout=0.1):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientLinearAttention(dim, num_heads, dropout)
        
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Pre-norm architecture
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
```

---

## 4. Module 3: Boundary Detection Head

### Semantic Boundary Attention

```python
class SemanticBoundaryAttention(nn.Module):
    """
    Attention mechanism specifically designed for boundary detection
    
    Key insight: Boundaries occur where semantic content changes
    We use attention weights directly as boundary indicators
    """
    
    def __init__(self, dim=256, num_heads=4):
        super().__init__()
        
        self.attention = EfficientLinearAttention(dim, num_heads)
        self.boundary_mlp = nn.Sequential(
            nn.Linear(dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Learnable boundary threshold
        self.threshold = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, temporal_features):
        # temporal_features: [B, T, 256]
        
        # Apply attention
        attended, attn_weights = self.attention(temporal_features)
        
        # Compute boundary probability at each timestep
        boundary_probs = self.boundary_mlp(attended)  # [B, T, 1]
        
        # Sharpen with learned threshold
        boundary_probs = torch.sigmoid(
            10 * (boundary_probs - self.threshold)
        )
        
        return boundary_probs, attn_weights
```

### Boundary Loss

```python
def boundary_detection_loss(pred_boundaries, true_boundaries, weights=None):
    """
    Weighted binary cross-entropy for boundary detection
    
    Boundaries are rare (typically <5% of timesteps), so we use:
    1. Class weighting to handle imbalance
    2. Focal loss to focus on hard examples
    """
    
    # Class weights (boundaries weighted higher)
    pos_weight = torch.tensor(10.0)  # Boundaries are 10x more important
    
    # Focal loss parameters
    gamma = 2.0
    
    # Binary cross-entropy with logits
    bce = F.binary_cross_entropy_with_logits(
        pred_boundaries, 
        true_boundaries,
        pos_weight=pos_weight,
        reduction='none'
    )
    
    # Focal weighting
    pt = torch.exp(-bce)
    focal_weight = (1 - pt) ** gamma
    
    loss = (focal_weight * bce).mean()
    
    return loss
```

---

## 5. Module 4: Transitional Activity Specialization

### Multi-Scale Transition Module

```python
class TransitionalActivityModule(nn.Module):
    """
    Specialized module for detecting brief, dynamic transitional activities
    
    Key challenges:
    1. Short duration (1-3 seconds)
    2. High dynamics (rapid acceleration changes)
    3. Variable patterns across users
    
    Solutions:
    1. Multi-scale temporal convolutions
    2. Dynamic variance attention
    3. Transition-specific features (derivatives)
    """
    
    def __init__(self, dim=256):
        super().__init__()
        
        # Multi-scale temporal convolutions
        self.scales = nn.ModuleList([
            nn.Conv1d(dim, 64, kernel_size=k, padding=k//2, groups=4)
            for k in [3, 5, 7, 11]  # Different temporal scales
        ])
        
        # Dynamic variance attention
        self.variance_attention = DynamicVarianceAttention(256)
        
        # Transition-specific features
        self.transition_features = TransitionFeatureExtractor()
        
        # Fusion
        self.fusion = nn.Linear(256 + 64 + 16, 64)
        
    def forward(self, x):
        # x: [B, T, 256]
        
        # Multi-scale features
        multi_scale = [conv(x.transpose(1, 2)) for conv in self.scales]
        multi_scale = torch.cat(multi_scale, dim=1)  # [B, 256, T]
        multi_scale = multi_scale.transpose(1, 2)  # [B, T, 256]
        
        # Dynamic attention on high-variance regions
        attended = self.variance_attention(multi_scale)  # [B, T, 256]
        
        # Transition-specific features (derivatives)
        trans_feats = self.transition_features(x)  # [B, T, 16]
        
        # Fuse all features
        combined = torch.cat([x, attended, trans_feats], dim=-1)
        output = self.fusion(combined)  # [B, T, 64]
        
        return output
```

### Dynamic Variance Attention

```python
class DynamicVarianceAttention(nn.Module):
    """
    Attention that emphasizes high-variance (dynamic) regions
    
    Transitional activities have high variance in acceleration
    This module learns to focus attention on these regions
    """
    
    def __init__(self, dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, 1),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        # x: [B, T, dim]
        
        # Compute attention weights
        attn_weights = self.attention(x)  # [B, T, 1]
        
        # Apply attention
        attended = x * attn_weights
        
        return attended
```

---

## 6. Module 5: Classification Head

```python
class ClassificationHead(nn.Module):
    """
    Activity classification from segmented features
    """
    
    def __init__(self, input_dim=320, hidden_dim=128, num_classes=6):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, x):
        # x: [B, T, 320] (256 base + 64 transition)
        
        # Global average pooling over time
        x = x.mean(dim=1)  # [B, 320]
        
        # Classify
        logits = self.classifier(x)  # [B, num_classes]
        
        return logits
```

---

## 7. Complete Model

```python
class SASHAR(nn.Module):
    """
    Self-Supervised Attention-based Segmentation for HAR
    
    Total Parameters: ~150K (before distillation)
    After Distillation: <25K
    """
    
    def __init__(self, num_classes=6):
        super().__init__()
        
        # Core modules
        self.encoder = CNNFeatureEncoder()
        self.temporal = nn.Sequential(*[
            TransformerBlock(256, num_heads=4)
            for _ in range(3)
        ])
        self.boundary_head = SemanticBoundaryAttention(256)
        self.transition_module = TransitionalActivityModule(256)
        self.classifier = ClassificationHead(320, 128, num_classes)
        
    def forward(self, x):
        # x: [B, 6, T]
        
        # Encode
        features = self.encoder(x)  # [B, 256, T']
        features = features.transpose(1, 2)  # [B, T', 256]
        
        # Temporal modeling
        temporal_features = self.temporal(features)  # [B, T', 256]
        
        # Boundary detection
        boundary_probs, attn_weights = self.boundary_head(temporal_features)
        
        # Transition specialization
        transition_features = self.transition_module(temporal_features)
        
        # Combine for classification
        combined = torch.cat([temporal_features, transition_features], dim=-1)
        
        # Classify
        logits = self.classifier(combined)
        
        return {
            'logits': logits,
            'boundaries': boundary_probs,
            'attention': attn_weights
        }
```

---

## 8. Training Procedure

### Phase 1: Self-Supervised Pre-training

```python
def self_supervised_pretrain(model, unlabeled_data, epochs=100):
    """
    Pre-train using TCBL on unlabeled data
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    for epoch in range(epochs):
        for batch in unlabeled_data:
            # Three pretext tasks
            loss_tc = temporal_contrastive_loss(model, batch)
            loss_cp = continuity_prediction_loss(model, batch)
            loss_mt = masked_temporal_loss(model, batch)
            
            # Combined loss
            total_loss = loss_tc + loss_cp + loss_mt
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
```

### Phase 2: Supervised Fine-tuning

```python
def supervised_finetune(model, labeled_data, epochs=50):
    """
    Fine-tune with activity and boundary labels
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    
    for epoch in range(epochs):
        for batch, activity_labels, boundary_labels in labeled_data:
            outputs = model(batch)
            
            # Multi-task loss
            loss_cls = F.cross_entropy(outputs['logits'], activity_labels)
            loss_seg = boundary_detection_loss(outputs['boundaries'], boundary_labels)
            
            # Combined
            total_loss = loss_cls + loss_seg
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
```

### Phase 3: Knowledge Distillation

```python
def distill_model(teacher, student, data, epochs=30):
    """
    Distill large teacher to small student
    """
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4)
    
    for epoch in range(epochs):
        for batch, labels in data:
            # Teacher outputs (no grad)
            with torch.no_grad():
                teacher_outputs = teacher(batch)
            
            # Student outputs
            student_outputs = student(batch)
            
            # Distillation loss
            loss = distillation_loss(
                student_outputs['logits'],
                teacher_outputs['logits'],
                labels,
                temperature=3.0
            )
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

---

*Last Updated: March 2026*
