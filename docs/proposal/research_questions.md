# Research Questions

## Primary Research Question

> **RQ0:** How can we develop a unified framework that achieves accurate temporal segmentation and activity recognition on resource-constrained edge devices without requiring extensive labeled training data?

---

## Secondary Research Questions

### RQ1: Self-Supervised Boundary Learning

**Question:** Can we design pretext tasks that enable learning of activity boundaries from unlabeled continuous sensor streams?

**Motivation:** 
- Annotation of activity boundaries is expensive and subjective
- Existing self-supervised methods focus on classification, not segmentation
- Need to reduce dependency on labeled data

**Hypothesis:** Contrastive learning between temporal segments combined with continuity prediction can discover meaningful boundaries without labels.

**Validation Criteria:**
- Achieve >90% of supervised performance with <10% labeled data
- Demonstrate transfer learning across datasets
- Show learned boundaries align with human annotations

---

### RQ2: Attention-Based Segmentation

**Question:** Do attention mechanisms provide advantages over similarity-based methods for temporal boundary detection?

**Motivation:**
- Statistical similarity methods (Baraka 2023) use hand-crafted features
- Deep similarity (Baraka 2024) uses CNNs with limited temporal context
- Transformers excel at long-range dependencies in other domains

**Hypothesis:** Self-attention captures long-range temporal dependencies that improve boundary detection, especially for gradual transitions.

**Validation Criteria:**
- +5% boundary F1 over Deep Similarity Segmentation
- Better performance on gradual transitions (sit-to-lie)
- Interpretable attention maps showing temporal focus

---

### RQ3: Efficient Architecture Design

**Question:** What is the optimal architecture for simultaneous segmentation and classification on edge devices?

**Motivation:**
- SOTA models have millions of parameters
- Edge devices have <100KB memory for ML
- Need joint optimization of both tasks

**Hypothesis:** A hybrid CNN-Transformer with knowledge distillation can achieve <25K parameters while maintaining >98% accuracy.

**Validation Criteria:**
- <25K parameters
- >98% classification accuracy
- <1ms inference latency on mobile CPU

---

### RQ4: Edge Optimization

**Question:** How can we optimize the complete segmentation-classification pipeline for nanojoule-level energy consumption?

**Motivation:**
- Wearables have strict battery constraints
- Existing efficient models focus only on classification
- Need end-to-end pipeline optimization

**Hypothesis:** Joint quantization-aware training and weightless neural network components can achieve <45 nJ/sample.

**Validation Criteria:**
- <45 nJ/sample energy consumption
- <1% accuracy loss from quantization
- Deployment on Arduino Nano / STM32

---

### RQ5: Transitional Activity Modeling

**Question:** Can specialized components improve transitional activity detection without sacrificing overall performance?

**Motivation:**
- Static activities: 95-99% F1
- Transitional activities: 70-85% F1
- 15-25 percentage point gap

**Hypothesis:** A transition-aware attention module can improve transitional F1 by >5% with <1% impact on static activities.

**Validation Criteria:**
- >94% F1 on transitional activities
- <1% degradation on static activities
- Demonstrated on sit-to-stand, stand-to-sit, sit-to-lie

---

## Research Question Dependencies

```
RQ0 (Primary)
├── RQ1 (Self-Supervised) ─────────┐
├── RQ2 (Attention) ───────────────┤
├── RQ3 (Architecture) ────────────┼──→ Integrated Framework
├── RQ4 (Edge Optimization) ───────┤
└── RQ5 (Transitional) ────────────┘
```

---

## Evaluation Protocol

| RQ | Primary Metric | Secondary Metrics | Target |
|----|---------------|-------------------|--------|
| RQ1 | Label Efficiency | Transfer Accuracy | 90% @ 10% labels |
| RQ2 | Boundary F1 | Attention Interpretability | >97% |
| RQ3 | Parameters | Accuracy, Latency | <25K, >98% |
| RQ4 | Energy (nJ) | Memory, Latency | <45 nJ |
| RQ5 | Transitional F1 | Overall F1 | >94% |

---

*Last Updated: March 2026*
