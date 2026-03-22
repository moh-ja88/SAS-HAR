# Experimental Plan

## Overview

This document outlines the comprehensive experimental protocol for evaluating the SAS-HAR framework.

---

## 1. Experimental Design

### 1.1 Research Questions → Experiments Mapping

| RQ | Experiment | Primary Metric | Datasets |
|----|------------|----------------|----------|
| RQ1 | Label Efficiency | Boundary F1 @ 10% labels | All |
| RQ2 | Attention vs. Similarity | Boundary F1 | All |
| RQ3 | Architecture Efficiency | Parameters vs. Accuracy | All |
| RQ4 | Edge Deployment | Energy (nJ) | All |
| RQ5 | Transitional Activities | Transitional F1 | PAMAP2 |

---

## 2. Experiment 1: Self-Supervised Learning Evaluation

### 2.1 Objective
Evaluate label efficiency of Temporal Contrastive Boundary Learning (TCBL).

### 2.2 Protocol

#### Pre-training
```
Dataset: All unlabeled data from WISDM, UCI-HAR, PAMAP2, Opportunity
Duration: 100 epochs
Batch Size: 64
Learning Rate: 1e-4 (Adam)
Pretext Tasks: TC + CP + MT
```

#### Fine-tuning
```
Labels: 1%, 5%, 10%, 25%, 50%, 100%
Duration: 50 epochs
Learning Rate: 1e-5 (Adam)
```

### 2.3 Evaluation

| Labels Used | Metric |
|-------------|--------|
| 1% | Boundary F1 |
| 5% | Boundary F1 |
| 10% | Boundary F1 |
| 25% | Boundary F1 |
| 50% | Boundary F1 |
| 100% | Boundary F1 |

### 2.4 Expected Results

| Labels | WISDM | UCI-HAR | PAMAP2 | Opportunity |
|--------|-------|---------|--------|-------------|
| 1% | 78% | 76% | 72% | 68% |
| 5% | 86% | 84% | 81% | 78% |
| 10% | 90% | 90% | 87% | 84% |
| 50% | 96% | 95% | 94% | 92% |
| 100% | 98% | 98% | 97% | 95% |

---

## 3. Experiment 2: Segmentation Accuracy

### 3.1 Objective
Compare SAS-HAR segmentation to state-of-the-art methods.

### 3.2 Baselines

| Method | Year | Type | Description |
|--------|------|------|-------------|
| FSW | Traditional | Fixed | 2.56s window, 50% overlap |
| ASW | 2017 | Adaptive | Energy-based sizing |
| SSS | 2023 | Similarity | Statistical features |
| DSS | 2024 | Similarity | CNN-based |
| P2LHAP | 2024 | Transformer | Patch-based |
| **SAS-HAR** | 2026 | Attention | Self-supervised |

### 3.3 Protocol

```
Training: Full supervision
Evaluation: Cross-subject
Metrics: Boundary F1, Segmentation Accuracy
Runs: 5 random seeds, report mean ± std
```

### 3.4 Expected Results

| Method | WISDM | UCI-HAR | PAMAP2 | Avg |
|--------|-------|---------|--------|-----|
| FSW | 72.3 | 71.8 | 68.5 | 70.9 |
| ASW | 78.5 | 77.2 | 74.1 | 76.6 |
| SSS | 83.5 | 82.8 | 79.2 | 81.8 |
| DSS | 89.1 | 88.5 | 85.7 | 87.8 |
| P2LHAP | 95.2 | 94.8 | 92.1 | 94.0 |
| **SAS-HAR** | **97.2** | **96.8** | **94.5** | **96.2** |

---

## 4. Experiment 3: Classification Performance

### 4.1 Objective
Evaluate classification accuracy with dynamic segmentation.

### 4.2 Protocol

```
Input: Raw sensor stream
Output: Activity labels
Metrics: Accuracy, Weighted F1
Comparison: Fixed window vs. Dynamic segmentation
```

### 4.3 Expected Results

| Method | WISDM | UCI-HAR | PAMAP2 |
|--------|-------|---------|--------|
| Fixed + CNN | 94.2 | 95.1 | 91.8 |
| Fixed + LSTM | 95.3 | 96.2 | 92.5 |
| Fixed + Transformer | 96.1 | 97.2 | 93.8 |
| DSS + CNN | 95.8 | 96.5 | 93.2 |
| **SAS-HAR (Dynamic)** | **98.1** | **98.5** | **96.8** |

---

## 5. Experiment 4: Transitional Activity Evaluation

### 5.1 Objective
Evaluate TASM module on transitional activities.

### 5.2 Protocol

```
Dataset: PAMAP2 (has transitional activities)
Evaluation: Per-activity F1 score
Focus: Transitional activities
Comparison: With vs. without TASM
```

### 5.3 Transitional Activities

- Walking → Running
- Running → Walking
- Lying → Sitting
- Sitting → Standing
- Standing → Walking

### 5.4 Expected Results

| Activity | Baseline | +TASM | Improvement |
|----------|----------|-------|-------------|
| Walk→Run | 82% | 95% | +13% |
| Run→Walk | 80% | 94% | +14% |
| Lie→Sit | 72% | 90% | +18% |
| Sit→Stand | 78% | 94% | +16% |
| Stand→Walk | 75% | 92% | +17% |
| **Average** | **77%** | **93%** | **+16%** |

---

## 6. Experiment 5: Edge Deployment

### 6.1 Objective
Evaluate efficiency metrics on real hardware.

### 6.2 Hardware Platforms

| Platform | CPU | Memory | Clock |
|----------|-----|--------|-------|
| Mobile (Snapdragon 865) | Kryo 585 | 8GB | 2.84 GHz |
| Arduino Nano 33 BLE | nRF52840 | 256KB | 64 MHz |
| STM32F4 Discovery | Cortex-M4 | 192KB | 168 MHz |

### 6.3 Protocol

1. **Model Preparation**
   - Train teacher model (SAS-HAR)
   - Distill to student (<25K params)
   - Quantize to INT8

2. **Deployment**
   - Convert to TensorFlow Lite
   - Optimize for target platform
   - Deploy and measure

3. **Measurement**
   - Energy: Power monitor
   - Latency: High-resolution timer
   - Memory: Runtime profiling

### 6.4 Expected Results

| Platform | Params | Energy (nJ) | Latency (ms) | Memory (KB) |
|----------|--------|-------------|--------------|-------------|
| Mobile | 24K | 12 | 0.8 | 32 |
| Arduino | 24K | 42 | 3.2 | 18 |
| STM32F4 | 24K | 38 | 2.8 | 18 |

---

## 7. Experiment 6: Ablation Studies

### 7.1 Objective
Determine contribution of each component.

### 7.2 Ablation Configurations

| Config | Components | Missing |
|--------|-----------|---------|
| Full | All | None |
| -TC | No temporal contrastive | TC pretext task |
| -CP | No continuity prediction | CP pretext task |
| -MT | No masked temporal | MT pretext task |
| -SBA | No boundary attention | SBA module |
| -TASM | No transition module | TASM |
| -KD | No knowledge distillation | Large model |

### 7.3 Expected Results

| Config | Boundary F1 | Accuracy | Impact |
|--------|-------------|----------|--------|
| Full | 97.2 | 98.3 | Baseline |
| -TC | 92.5 | 96.1 | -4.7% |
| -CP | 94.8 | 97.2 | -2.4% |
| -MT | 96.1 | 97.8 | -1.1% |
| -SBA | 92.8 | 95.5 | -4.4% |
| -TASM | 94.2 | 96.8 | -3.0% |
| -KD | 97.0 | 98.1 | +600K params |

---

## 8. Experiment 7: Real-World Validation

### 8.1 Objective
Validate performance in unconstrained real-world conditions.

### 8.2 User Study Design

```
Participants: 10 (5 male, 5 female)
Ages: 25-65
Duration: 2 weeks per participant
Device: Arduino Nano 33 BLE (wrist)
Protocol: Unconstrained daily activities
Ground Truth: Self-report + spot-checks
```

### 8.3 Evaluation Metrics

- Accuracy degradation vs. lab
- User satisfaction (1-5 scale)
- Comfort (1-5 scale)
- Battery life (hours)
- False alarm rate

### 8.4 Expected Results

| Metric | Lab | Real-World | Degradation |
|--------|-----|------------|-------------|
| Accuracy | 97.5% | 94.8% | -2.7% |
| Boundary F1 | 96.5% | 92.1% | -4.4% |
| User Satisfaction | - | 4.2/5 | - |
| Battery Life | - | 72h | - |

---

## 9. Statistical Analysis

### 9.1 Significance Testing

- **Test**: Paired t-test (for paired comparisons) or Wilcoxon signed-rank
- **Significance Level**: p < 0.05
- **Multiple Comparisons**: Bonferroni correction

### 9.2 Confidence Intervals

- Report: Mean ± 95% CI
- Based on: 5 random seeds

### 9.3 Effect Size

- **Metric**: Cohen's d
- **Interpretation**: Small (0.2), Medium (0.5), Large (0.8)

---

## 10. Reproducibility

### 10.1 Code Availability

- GitHub repository with all code
- Requirements.txt with exact versions
- Random seeds documented
- Configuration files for all experiments

### 10.2 Data Availability

- Public dataset download scripts
- Preprocessing code
- Train/test splits

### 10.3 Model Availability

- Pre-trained weights
- Training logs
- Hyperparameter configurations

---

## 11. Timeline

| Month | Experiment | Duration |
|-------|------------|----------|
| 13-15 | Exp 1: SSL Evaluation | 3 months |
| 16-18 | Exp 2: Segmentation | 3 months |
| 19-21 | Exp 3-4: Classification + Transitions | 3 months |
| 22-24 | Exp 5-6: Edge + Ablation | 3 months |
| 31-33 | Exp 7: Real-World | 3 months |

---

*Last Updated: March 2026*
