# Research Gap Analysis

## Executive Summary

This document identifies critical research gaps in Human Activity Recognition (HAR) with focus on temporal segmentation, self-supervised learning, and edge deployment. The analysis forms the foundation for our SAS-HAR research contributions.

## Gap Identification Framework

```
┌─────────────────────────────────────────────────────────┐
│                  Research Gap Analysis                   │
├─────────────────────────────────────────────────────────┤
│  1. What exists? (SOTA Review)                          │
│  2. What's missing? (Gap Identification)                │
│  3. Why does it matter? (Impact Assessment)             │
│  4. How to address? (Our Contribution)                  │
└─────────────────────────────────────────────────────────┘
```

## Gap 1: Self-Supervised Temporal Segmentation

### Current State (2024-2025)

| Method | Supervision | Boundary Detection | Novel Activities |
|--------|-------------|-------------------|------------------|
| Baraka 2024 | Supervised | Similarity-based | Poor |
| P2LHAP 2024 | Supervised | Attention-based | Moderate |
| Adaptive Window 2017 | Unsupervised | Statistical | Good |
| TinierHAR 2024 | Supervised | None | N/A |

### The Gap

**No existing method combines:**
1. Self-supervised learning for segmentation
2. High boundary detection accuracy (>90% F1)
3. Generalization to unseen activities
4. Real-time edge deployment

### Evidence

- Baraka et al. (2024): Requires labeled boundaries
- P2LHAP: Supervised prompts, limited transfer
- Traditional methods (CUSUM, sliding window): Low accuracy

### Impact Assessment

| Criterion | Score | Justification |
|-----------|-------|---------------|
| Research Value | 9.5/10 | Untouched research area |
| Practical Impact | 9.0/10 | Enables label-efficient HAR |
| Feasibility | 8.0/10 | Self-supervised learning mature |

### Our Solution: TCBL Framework

**Temporal Contrastive Boundary Learning (TCBL)**

```
┌─────────────────────────────────────────┐
│           TCBL Framework                │
├─────────────────────────────────────────┤
│  Pre-training (Unlabeled Data):         │
│  • Contrastive learning at boundaries   │
│  • Temporal consistency objectives      │
│  • Activity-aware augmentations         │
│                                         │
│  Fine-tuning (Limited Labels):          │
│  • Boundary detection head              │
│  • Activity classification head         │
└─────────────────────────────────────────┘
```

**Novelty Score: 9.5/10**

---

## Gap 2: Unified Segmentation-Recognition Framework

### Current State

Existing systems treat segmentation and recognition as separate:

```
Traditional Pipeline:
Sensor → Window → Features → Classification
         ↓
    Segmentation (separate module)

Problems:
- Error propagation
- Suboptimal boundaries for classification
- No shared representations
```

### The Gap

**No unified framework that:**
1. Jointly optimizes segmentation and recognition
2. Shares representations between tasks
3. Handles variable-length activities
4. Maintains real-time performance

### Evidence

| Method | Unified? | Shared Rep? | Real-time? |
|--------|----------|-------------|------------|
| DeepConvLSTM | No | N/A | Yes |
| Deep Similarity | Partial | No | Moderate |
| P2LHAP | Partial | Limited | No |

### Impact Assessment

| Criterion | Score | Justification |
|-----------|-------|---------------|
| Research Value | 8.5/10 | Multi-task learning in HAR |
| Practical Impact | 9.0/10 | Better accuracy, simpler deployment |
| Feasibility | 9.0/10 | Multi-task learning well-established |

### Our Solution: SAS-HAR Architecture

```
┌─────────────────────────────────────────────────────┐
│              SAS-HAR Unified Framework              │
├─────────────────────────────────────────────────────┤
│                                                     │
│  Raw Sensor Data                                    │
│         │                                           │
│         ▼                                           │
│  ┌─────────────┐                                    │
│  │ CNN Encoder │ ← Shared Feature Extraction        │
│  └──────┬──────┘                                    │
│         │                                           │
│         ▼                                           │
│  ┌─────────────────┐                                │
│  │ Transformer     │ ← Temporal Context             │
│  │ Encoder         │                                 │
│  └──────┬──────────┘                                │
│         │                                           │
│    ┌────┴────┐                                      │
│    ▼         ▼                                      │
│ Boundary   Activity                                 │
│  Head     Classifier                                │
│    │         │                                      │
│    └────┬────┘                                      │
│         ▼                                           │
│    Joint Loss                                       │
│                                                     │
└─────────────────────────────────────────────────────┘
```

**Novelty Score: 8.5/10**

---

## Gap 3: Attention-Based Boundary Detection

### Current State

| Method | Boundary Mechanism | Accuracy | Interpretability |
|--------|-------------------|----------|------------------|
| Sliding Window | Fixed | Low | High |
| Similarity Segmentation | Threshold | Moderate | Moderate |
| Change Point Detection | Statistical | Moderate | High |
| Deep Similarity | Learned | Good | Low |

### The Gap

**Attention mechanisms underutilized for:**
1. Explicit boundary localization
2. Interpretable transition detection
3. Adaptive focus on relevant time steps
4. Multi-scale boundary analysis

### Evidence

- P2LHAP uses attention but not explicitly for boundaries
- Existing methods use post-hoc attention visualization
- No attention-based boundary loss functions

### Impact Assessment

| Criterion | Score | Justification |
|-----------|-------|---------------|
| Research Value | 8.5/10 | Novel attention application |
| Practical Impact | 8.0/10 | Better boundaries, interpretability |
| Feasibility | 9.0/10 | Attention mechanisms mature |

### Our Solution: Boundary Attention Module

```python
class BoundaryAttention(nn.Module):
    """
    Learnable attention for boundary detection
    """
    def __init__(self, d_model, n_heads):
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.boundary_query = nn.Parameter(torch.randn(1, d_model))
    
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        # Boundary query attends to all positions
        boundary_scores = self.attention(
            query=self.boundary_query.expand(x.size(0), -1, -1),
            key=x, value=x
        )
        return boundary_scores
```

**Novelty Score: 8.5/10**

---

## Gap 4: Edge-Optimized Segmentation Models

### Current State (2024-2025)

| Model | Parameters | Energy | Latency | Segmentation? |
|-------|------------|--------|---------|---------------|
| TinierHAR | 34K | ~200 nJ | <5ms | No |
| nanoML-HAR | 25K | 56-104 nJ | <3ms | No |
| MCUNet | 47K | ~150 nJ | ~5ms | No |

### The Gap

**No existing edge-optimized model for:**
1. Real-time segmentation on microcontrollers
2. Sub-50K parameters with segmentation capability
3. Energy-efficient boundary detection
4. Memory-constrained deployment

### Evidence

- TinierHAR: Classification only
- nanoML: No temporal segmentation
- All lightweight HAR models: Single activity per window

### Impact Assessment

| Criterion | Score | Justification |
|-----------|-------|---------------|
| Research Value | 8.0/10 | Applied research gap |
| Practical Impact | 9.5/10 | Enables wearable deployment |
| Feasibility | 8.0/10 | Requires optimization expertise |

### Our Solution: EdgeSAS-HAR

```
┌─────────────────────────────────────────────────────┐
│           EdgeSAS-HAR Optimization                  │
├─────────────────────────────────────────────────────┤
│                                                     │
│  1. Knowledge Distillation                          │
│     Teacher (Full SAS-HAR) → Student (EdgeSAS)     │
│                                                     │
│  2. Quantization-Aware Training                     │
│     FP32 → INT8 with accuracy preservation         │
│                                                     │
│  3. Neural Architecture Search                      │
│     Automated tiny model discovery                  │
│                                                     │
│  4. Efficient Attention                             │
│     Linear attention for O(n) complexity           │
│                                                     │
└─────────────────────────────────────────────────────┘
```

**Target Specifications:**
- Parameters: <50K
- Energy: <100 nJ/sample
- Latency: <5ms
- Boundary F1: >85%

**Novelty Score: 8.0/10**

---

## Gap 5: Transitional Activity Handling

### Current State

| Method | Transitional Handling | Accuracy on Transitions |
|--------|----------------------|------------------------|
| Standard HAR | Ignore | N/A |
| Sliding Window | Arbitrary boundaries | ~60% |
| Similarity Segmentation | Threshold-based | ~70% |
| P2LHAP | Prompt learning | ~85% |

### The Gap

**Transitional activities (walking→running, sit→stand) are:**
1. Poorly defined in most datasets
2. Difficult to segment accurately
3. Critical for real-world applications
4. Under-researched in HAR literature

### Evidence

- UCI-HAR: No transitional labels
- WISDM: Limited transition samples
- PAMAP2: Some transitions but sparse
- Papers rarely report transition accuracy separately

### Impact Assessment

| Criterion | Score | Justification |
|-----------|-------|---------------|
| Research Value | 8.5/10 | Underexplored problem |
| Practical Impact | 9.0/10 | Real-world importance |
| Feasibility | 7.5/10 | Dataset limitations |

### Our Solution: Transition-Aware Loss

```python
class TransitionAwareLoss(nn.Module):
    """
    Emphasizes transitional activity boundaries
    """
    def __init__(self, transition_weight=2.0):
        self.transition_weight = transition_weight
    
    def forward(self, pred, target, is_transition):
        # Standard cross-entropy
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        
        # Weight transitions higher
        weights = torch.where(
            is_transition,
            self.transition_weight * torch.ones_like(ce_loss),
            torch.ones_like(ce_loss)
        )
        
        return (ce_loss * weights).mean()
```

**Novelty Score: 8.5/10**

---

## Summary: Gap Prioritization

| Gap | Novelty | Impact | Feasibility | Priority |
|-----|---------|--------|-------------|----------|
| 1. Self-Supervised Segmentation | 9.5 | 9.0 | 8.0 | **Highest** |
| 2. Unified Framework | 8.5 | 9.0 | 9.0 | High |
| 3. Attention-Based Boundaries | 8.5 | 8.0 | 9.0 | High |
| 4. Edge Optimization | 8.0 | 9.5 | 8.0 | Medium |
| 5. Transitional Activities | 8.5 | 9.0 | 7.5 | Medium |

## Research Contribution Mapping

```
Gap 1 (Self-Supervised)     → Paper 1: Dynamic Segmentation
Gap 2 (Unified Framework)   → Core: SAS-HAR Architecture
Gap 3 (Attention)           → Core: SAS-HAR Architecture
Gap 4 (Edge)                → Paper 3: Edge AI Framework
Gap 5 (Transitions)         → Paper 1: Dynamic Segmentation
```

## Comparison with Existing Work

| Our Contribution | Most Similar Work | Key Differentiator |
|-----------------|-------------------|-------------------|
| TCBL | Baraka 2024 | Self-supervised vs supervised |
| Unified SAS-HAR | DeepConvLSTM | Integrated segmentation |
| Boundary Attention | P2LHAP | Explicit boundary focus |
| EdgeSAS | TinierHAR | Segmentation capability |
| Transition Loss | Adaptive Window 2017 | Deep learning approach |

## Risk Assessment

### Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Self-supervised fails to converge | Medium | High | Pre-training curriculum |
| Edge model too slow | Low | Medium | Efficient attention variants |
| Transition labels unavailable | High | Medium | Weak supervision, synthesis |

### Research Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| SOTA advances before completion | Medium | Medium | Focus on unique contributions |
| Reviewers question novelty | Low | High | Thorough literature review |
| Datasets insufficient | Low | Medium | Collect supplementary data |

## References

1. Baraka, A., et al. (2024). "Deep Similarity Segmentation Model for Sensor-Based Activity Recognition."
2. Liu, H., et al. (2024). "TinierHAR: A Lightweight Deep Learning Architecture."
3. Zhang, H., et al. (2024). "P2LHAP: Prompt-to-Learn Human Activity Prediction."
4. Noor, M.H.M., et al. (2017). "Adaptive Sliding Window Segmentation."
5. Recent surveys (2024): IEEE Sensors, ACM Computing Surveys

---

**Document Version:** 1.0  
**Last Updated:** 2025  
**Authors:** PhD Research Team
