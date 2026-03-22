# Research Gap Analysis

## Executive Summary

This document identifies critical research gaps in Human Activity Recognition (HAR) that motivate our proposed research. Based on comprehensive literature review (2020-2026), we identify **five major gaps** that represent opportunities for significant contributions.

---

## Gap 1: Self-Supervised Temporal Segmentation

### Current State

| Aspect | Status | Evidence |
|--------|--------|----------|
| SSL for Classification | ✅ Well-established | SimCLR, MoCo, BYOL applied to HAR |
| SSL for Segmentation | ❌ Not addressed | No methods for unsupervised boundaries |
| Evaluation | ⚠️ Requires labels | Need labeled boundaries to evaluate |

**Key Finding:** While self-supervised learning has revolutionized HAR classification, **no existing method addresses temporal segmentation in a purely self-supervised manner.**

### Why This Gap Exists

1. **Pretext Task Design:** Most SSL pretext tasks (contrastive, masked modeling) designed for classification, not boundary detection
2. **Evaluation Challenge:** How to evaluate boundaries without ground truth?
3. **Continuous Stream Processing:** SSL typically operates on fixed windows, not continuous streams

### Opportunity

Design novel pretext tasks that explicitly encourage boundary discovery:
- **Temporal Contrastive Learning:** Contrast segments across time
- **Continuity Prediction:** Predict if adjacent segments belong to same activity
- **Masked Temporal Modeling:** Mask time steps, predict from context

**Novelty Potential: 9.5/10**

---

## Gap 2: Attention-Based Boundary Detection

### Current State

| Method | Year | Approach | Boundary F1 |
|--------|------|----------|-------------|
| Fixed Sliding Window | Traditional | None | 70-75% |
| Adaptive Sliding Window | 2017 | Energy-based | 78-82% |
| Statistical Similarity | 2023 | Hand-crafted | 82-86% |
| Deep Similarity | 2024 | CNN-based | 85-89% |
| P2LHAP | 2024 | Patch Transformer | 93-95% |
| **Proposed (SAS-HAR)** | 2025 | Attention + SSL | **Target: >97%** |

**Key Finding:** Attention mechanisms show promise (P2LHAP 2024), but no method specifically targets boundary detection with attention.

### Why This Gap Exists

1. **Classification Focus:** Most transformer HAR papers focus on classification accuracy
2. **Patch-Based Processing:** P2LHAP uses patches but doesn't explicitly model boundaries
3. **Computational Cost:** Full attention is expensive; efficient variants not explored for boundaries

### Opportunity

Leverage attention for boundary-specific tasks:
- **Semantic Attention:** Attend to semantically meaningful regions
- **Boundary-Aware Attention:** Attention weights as boundary indicators
- **Efficient Linear Attention:** O(n) complexity for real-time processing

**Novelty Potential: 8.5/10**

---

## Gap 3: Edge Deployment of Segmentation

### Current State

| System | Focus | Parameters | Energy | Segmentation? |
|--------|-------|------------|--------|---------------|
| nanoML (2025) | Classification | Ultra-small | 56-104 nJ | ❌ |
| TinierHAR (2025) | Classification | 34K | - | ❌ |
| μBi-ConvLSTM (2026) | Classification | 11.4K | - | ❌ |
| Deep Similarity (2024) | Segmentation | 650K | High | ✅ |
| **Proposed (SAS-HAR)** | Both | <25K | <45 nJ | ✅ |

**Key Finding:** No existing framework optimizes the **complete segmentation-classification pipeline** for edge deployment.

### Why This Gap Exists

1. **Separate Optimization:** Segmentation and classification optimized separately
2. **Accuracy-Efficiency Trade-off:** High segmentation accuracy requires large models
3. **Hardware Constraints:** Segmentation adds computational overhead

### Opportunity

Joint optimization for efficiency:
- **Shared Representations:** Features used for both tasks
- **Knowledge Distillation:** Large teacher → small student
- **Quantization-Aware Training:** INT8 optimization

**Novelty Potential: 8.0/10**

---

## Gap 4: Transitional Activity Detection

### Current State

| Activity Type | Typical F1 | Challenge |
|---------------|-----------|-----------|
| Static (sitting, standing) | 95-99% | Low variability |
| Dynamic (walking, running) | 92-97% | Duration variability |
| **Transitional (sit-to-stand)** | **70-85%** | Short + variable + dynamic |

**Key Finding:** A **15-25 percentage point gap** exists between static and transitional activities.

### Why This Gap Exists

1. **Short Duration:** 1-3 seconds, limited samples for feature extraction
2. **High Variability:** Speed and style vary significantly across users
3. **Dataset Imbalance:** Transitions are minority class in most datasets
4. **Boundary Ambiguity:** Gradual transitions lack clear boundaries

### Opportunity

Specialized modeling for transitions:
- **Transition-Aware Attention:** Focus on brief, dynamic periods
- **Temporal Pyramids:** Multi-scale processing for variable durations
- **Synthetic Augmentation:** Generate transition examples

**Novelty Potential: 8.5/10**

---

## Gap 5: Real-World Deployment Validation

### Current State

| Evaluation Type | % of Papers | Issue |
|-----------------|-------------|-------|
| Lab Dataset Only | 85% | Controlled conditions |
| Real-World Deployment | 10% | Limited duration/users |
| Longitudinal Study | 5% | Expensive, rare |

**Key Finding:** Most HAR research evaluated on controlled lab datasets; **real-world deployment gap of 5-10% accuracy**.

### Why This Gap Exists

1. **Dataset Availability:** Public datasets are lab-based
2. **Annotation Difficulty:** Real-world annotation is expensive
3. **Ethical Concerns:** Privacy, consent for real-world data
4. **Publication Pressure:** Lab results easier to publish

### Opportunity

Rigorous real-world validation:
- **Longitudinal Deployment:** 2-4 weeks, 10+ users
- **In-the-Wild Metrics:** Beyond accuracy (battery, user satisfaction)
- **Open Dataset Release:** Contribute real-world benchmark

**Novelty Potential: 7.0/10** (validation contribution)

---

## Gap Summary Matrix

| Gap | Current SOTA | Our Target | Improvement | Feasibility |
|-----|-------------|------------|-------------|-------------|
| Self-Supervised Segmentation | Not addressed | 90% @ 10% labels | N/A | High |
| Attention Boundaries | 95% F1 | >97% F1 | +2% | High |
| Edge Segmentation | 34K (classification only) | <25K (full pipeline) | -26% | Medium |
| Transitional Activities | 70-85% F1 | >94% F1 | +9-24% | Medium |
| Real-World Validation | 5-10% gap | <5% gap | -50% | Medium |

---

## Strategic Positioning

### Our Approach

We address **four gaps simultaneously** through an integrated framework:

```
┌─────────────────────────────────────────────────────────┐
│                    SAS-HAR Framework                     │
├─────────────────────────────────────────────────────────┤
│  Gap 1: Self-Supervised Segmentation                    │
│         └── Temporal Contrastive Boundary Learning      │
│                                                          │
│  Gap 2: Attention-Based Boundaries                      │
│         └── Semantic Attention Module                   │
│                                                          │
│  Gap 3: Edge Deployment                                 │
│         └── Joint Optimization + Knowledge Distillation │
│                                                          │
│  Gap 4: Transitional Activities                         │
│         └── Transition-Aware Attention                  │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
              Gap 5: Real-World Validation
              └── Longitudinal deployment study
```

### Competitive Advantage

| Competitor | Gap 1 | Gap 2 | Gap 3 | Gap 4 | Gap 5 |
|------------|-------|-------|-------|-------|-------|
| P2LHAP (2024) | ❌ | ⚠️ | ❌ | ❌ | ❌ |
| TinierHAR (2025) | ❌ | ❌ | ✅ | ❌ | ❌ |
| Deep Similarity (2024) | ❌ | ❌ | ❌ | ⚠️ | ❌ |
| XTinyHAR (2025) | ❌ | ⚠️ | ✅ | ❌ | ❌ |
| **SAS-HAR (Proposed)** | ✅ | ✅ | ✅ | ✅ | ✅ |

---

## Conclusion

The identified gaps represent **significant opportunities** for PhD-level contributions:

1. **Gap 1** is completely unexplored → High novelty potential
2. **Gap 2** has recent interest but room for improvement
3. **Gap 3** is practical and impactful for deployment
4. **Gap 4** addresses a well-known, unsolved problem
5. **Gap 5** strengthens real-world applicability

By addressing these gaps in an integrated framework, we can achieve a **novelty score of 9.0/10** while producing practical, deployable results.

---

*Last Updated: March 2026*
