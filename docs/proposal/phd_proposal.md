# PhD Research Proposal

## Self-Supervised Attention-Based Temporal Segmentation for Human Activity Recognition on Edge Devices

---

**PhD Candidate:** Mohammed Jasim  
**Supervisor:** Dr. Mohd Halim Mohd Noor  
**Co-Supervisor:** [To be determined]  
**Institution:** Universiti Sains Malaysia  
**School:** School of Electrical and Electronic Engineering  
**Proposed Start Date:** January 2025  
**Expected Duration:** 4 Years (48 Months)

---

## Table of Contents

1. [Introduction and Motivation](#1-introduction-and-motivation)
2. [Problem Statement](#2-problem-statement)
3. [Research Gap Analysis](#3-research-gap-analysis)
4. [Research Questions](#4-research-questions)
5. [Research Objectives](#5-research-objectives)
6. [Proposed Methodology](#6-proposed-methodology)
7. [Novel Contributions](#7-novel-contributions)
8. [Experimental Plan](#8-experimental-plan)
9. [Expected Impact](#9-expected-impact)
10. [Research Timeline](#10-research-timeline)
11. [Expected Publications](#11-expected-publications)
12. [References](#12-references)

---

## 1. Introduction and Motivation

### 1.1 Background

Human Activity Recognition (HAR) has emerged as a cornerstone technology for ambient assisted living (AAL), healthcare monitoring, and preventive medicine [1, 2]. With the global population aging rapidly—projected to reach 1.5 billion people over 65 by 2050 [3]—there is an urgent need for intelligent systems that can autonomously monitor daily activities and detect anomalies such as falls, gait irregularities, and mobility decline [4, 5].

Wearable sensors, particularly inertial measurement units (IMUs) comprising accelerometers and gyroscopes, have become ubiquitous in smartphones and smartwatches, enabling continuous, unobtrusive monitoring of human movement [6]. However, despite significant advances in sensor technology and machine learning, HAR systems still face fundamental challenges that limit their real-world deployment.

### 1.2 The Segmentation Bottleneck

The vast majority of HAR systems rely on **fixed-size sliding window (FSW)** segmentation [7], which divides continuous sensor streams into predefined temporal windows (typically 2-5 seconds). While simple to implement, FSW suffers from three critical limitations:

1. **Semantic Ambiguity**: Fixed windows often cut across activity boundaries, creating segments containing multiple activities [8]
2. **Duration Mismatch**: Human activities have variable durations—transitional activities (sit-to-stand) complete in 1-2 seconds, while others (walking) can continue indefinitely [9]
3. **Transitional Activity Failure**: Short, dynamic transitions between static activities are poorly captured, leading to misclassification rates exceeding 30% for these critical movements [10]

### 1.3 The Annotation Bottleneck

State-of-the-art HAR models, particularly those based on transformers [11, 12], require substantial labeled training data. However, annotating sensor data is:
- **Expensive**: Expert annotation costs $50-100 per hour of data
- **Time-consuming**: Annotation takes 5-10x real-time
- **Error-prone**: Exact activity boundaries are subjective

This creates a fundamental tension: we need more data to train better models, but labeling is prohibitively expensive.

### 1.4 The Deployment Gap

Modern HAR models achieving >98% accuracy [13, 14] are often:
- **Computationally expensive**: Millions of parameters, billions of FLOPs
- **Energy-hungry**: Unsuitable for battery-powered wearables
- **Privacy-concerning**: Often require cloud processing of sensitive data

This creates a deployment gap where high-accuracy models cannot be deployed on the devices where they are most needed.

### 1.5 Motivation

This research is motivated by a simple observation: **the future of HAR lies in intelligent, efficient, and autonomous systems that can operate entirely on resource-constrained devices without requiring extensive labeled data.**

We envision a system that:
- Automatically discovers activity boundaries without fixed windows
- Learns meaningful representations from unlabeled sensor data
- Operates efficiently on wearable devices with nanojoule-level energy consumption
- Preserves user privacy by processing all data locally

---

## 2. Problem Statement

### 2.1 Core Research Problem

Current HAR systems face a **trilemma** of competing requirements:

```
                    ACCURACY
                       /\
                      /  \
                     /    \
                    /      \
                   /________\
              EFFICIENCY    PRIVACY
```

- **High accuracy** requires complex models and large labeled datasets
- **High efficiency** requires simple models with limited capacity
- **Privacy preservation** requires on-device processing without cloud access

Existing systems typically optimize for one or two vertices of this trilemma, but achieving all three simultaneously remains an open challenge.

### 2.2 Specific Technical Challenges

#### Challenge 1: Dynamic Segmentation
Fixed-size sliding windows impose arbitrary temporal boundaries that don't align with natural activity durations. **How can we automatically discover activity boundaries based on semantic content rather than fixed time intervals?**

#### Challenge 2: Label-Efficient Learning
State-of-the-art HAR models require thousands of labeled examples. **How can we learn meaningful activity representations from unlabeled or weakly-labeled sensor data?**

#### Challenge 3: Edge Deployment
Complex models achieving >98% accuracy cannot run on resource-constrained wearables. **How can we compress these models while maintaining accuracy for critical activities?**

#### Challenge 4: Transitional Activity Detection
Transitional activities (sit-to-stand, stand-to-sit) are brief, variable, and poorly captured by existing methods. **How can we design models specifically optimized for these critical movements?**

### 2.3 Research Foundation

This research builds on three foundational papers from our research group:

1. **Noor et al. (2017)** [15]: Adaptive sliding window based on signal energy
   - *Limitation:* Heuristic-based, requires manual threshold tuning

2. **Baraka & Mohd Noor (2023)** [16]: Statistical similarity segmentation
   - *Limitation:* Hand-crafted features, limited representational power

3. **Baraka & Mohd Noor (2024)** [17]: Deep similarity segmentation
   - *Limitation:* Fully supervised, no edge optimization

**Our goal is to extend this research line by introducing self-supervised learning, attention mechanisms, and edge AI optimization in a unified framework.**

---

## 3. Research Gap Analysis

### 3.1 Gap 1: Self-Supervised Temporal Segmentation

**Current State:**
- Self-supervised learning (SSL) widely used for classification [18, 19]
- Limited application to temporal segmentation
- Most SSL methods require labeled boundaries for evaluation

**Gap:** No existing method learns activity boundaries in a purely self-supervised manner from continuous sensor streams.

**Opportunity:** Design novel pretext tasks that encourage boundary discovery without labels.

### 3.2 Gap 2: Attention-Based Boundary Detection

**Current State:**
- Transformers achieve state-of-the-art in NLP and vision [20, 21]
- Recent HAR transformers focus on classification, not segmentation [22, 23]
- P2LHAP (2024) uses patch-based approach but requires full supervision [24]

**Gap:** No transformer-based method specifically designed for temporal boundary detection in sensor streams.

**Opportunity:** Leverage attention mechanisms to model long-range dependencies for boundary detection.

### 3.3 Gap 3: Efficient Edge Deployment with Segmentation

**Current State:**
- nanoML methods achieve 56-104 nJ/sample [25]
- TinierHAR achieves 34K parameters [26]
- But these focus on classification, not segmentation

**Gap:** No existing framework optimizes the entire segmentation-classification pipeline for edge deployment.

**Opportunity:** Joint optimization of segmentation and classification for maximum efficiency.

### 3.4 Gap 4: Transitional Activity Specialization

**Current State:**
- Static activities (sitting, standing): 95-99% F1
- Transitional activities (sit-to-stand): 70-85% F1
- Gap of 15-25 percentage points

**Gap:** No method specifically designed to excel at transitional activity detection while maintaining overall performance.

**Opportunity:** Specialized architecture components for transition modeling.

### 3.5 Summary of Research Gaps

| Gap | Current SOTA | Target | Novelty Potential |
|-----|-------------|--------|-------------------|
| Self-Supervised Segmentation | Not addressed | Fully unsupervised boundaries | 9.5/10 |
| Attention-Based Boundaries | P2LHAP (2024) | Semantic attention | 8.5/10 |
| Edge Segmentation | Classification only | Full pipeline | 8.0/10 |
| Transitional Activities | 70-85% F1 | >94% F1 | 8.5/10 |

---

## 4. Research Questions

### 4.1 Primary Research Question

> **RQ0:** How can we develop a unified framework that achieves accurate temporal segmentation and activity recognition on resource-constrained edge devices without requiring extensive labeled training data?

### 4.2 Secondary Research Questions

#### RQ1: Self-Supervised Boundary Learning
> Can we design pretext tasks that enable learning of activity boundaries from unlabeled continuous sensor streams?

**Hypothesis:** Contrastive learning between temporal segments combined with continuity prediction can discover meaningful boundaries without labels.

#### RQ2: Attention-Based Segmentation
> Do attention mechanisms provide advantages over similarity-based methods for temporal boundary detection?

**Hypothesis:** Self-attention captures long-range temporal dependencies that improve boundary detection, especially for gradual transitions.

#### RQ3: Efficient Architecture Design
> What is the optimal architecture for simultaneous segmentation and classification on edge devices?

**Hypothesis:** A hybrid CNN-Transformer with knowledge distillation can achieve <25K parameters while maintaining >98% accuracy.

#### RQ4: Edge Optimization
> How can we optimize the complete segmentation-classification pipeline for nanojoule-level energy consumption?

**Hypothesis:** Joint quantization-aware training and weightless neural network components can achieve <45 nJ/sample.

#### RQ5: Transitional Activity Modeling
> Can specialized components improve transitional activity detection without sacrificing overall performance?

**Hypothesis:** A transition-aware attention module can improve transitional F1 by >5% with <1% impact on static activities.

---

## 5. Research Objectives

### 5.1 Main Objective

To design, implement, and validate a **Self-Supervised Attention-based Similarity Segmentation (SAS-HAR)** framework that enables accurate, efficient, and privacy-preserving human activity recognition on resource-constrained edge devices.

### 5.2 Specific Objectives

#### Objective 1: Develop Self-Supervised Segmentation Algorithm
- Design novel pretext tasks for boundary learning
- Implement temporal contrastive learning framework
- Achieve >90% of supervised performance with <10% labels

#### Objective 2: Design Efficient Hybrid Architecture
- Create CNN-Transformer hybrid for spatial-temporal modeling
- Implement attention-based boundary detection module
- Target <25K parameters, >98% accuracy

#### Objective 3: Optimize for Edge Deployment
- Implement knowledge distillation from large teacher models
- Apply quantization-aware training (INT8)
- Target <45 nJ/sample energy, <1 ms latency

#### Objective 4: Validate on Real-World Deployment
- Deploy on actual wearable devices (smartwatch, microcontroller)
- Conduct user study with 10+ participants
- Demonstrate real-world robustness

---

## 6. Proposed Methodology

### 6.1 System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    SAS-HAR Framework                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────────┐   │
│  │   Sensor    │───▶│    CNN       │───▶│   Transformer    │   │
│  │   Stream    │    │   Encoder    │    │   Temporal Module│   │
│  └─────────────┘    └──────────────┘    └──────────────────┘   │
│                            │                      │              │
│                            ▼                      ▼              │
│                    ┌──────────────┐    ┌──────────────────┐    │
│                    │   Feature    │    │   Boundary       │    │
│                    │   Extractor  │    │   Detection      │    │
│                    └──────────────┘    └──────────────────┘    │
│                            │                      │              │
│                            └──────────┬───────────┘              │
│                                       ▼                          │
│                           ┌──────────────────┐                  │
│                           │   Activity       │                  │
│                           │   Classifier     │                  │
│                           └──────────────────┘                  │
│                                       │                          │
│  ┌─────────────────────────────────────┴────────────────────┐   │
│  │               Self-Supervised Learning Module             │   │
│  │  ┌─────────────────┐  ┌─────────────────┐               │   │
│  │  │  Temporal       │  │  Continuity     │               │   │
│  │  │  Contrastive    │  │  Prediction     │               │   │
│  │  └─────────────────┘  └─────────────────┘               │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                       │                          │
│                                       ▼                          │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    Edge Deployment Layer                  │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │   │
│  │  │ Knowledge   │  │ Quantization│  │ Weightless NN   │  │   │
│  │  │ Distillation│  │ (INT8)      │  │ (Optional)      │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────────┘  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 Component 1: CNN Feature Encoder

**Purpose:** Extract local spatial features from raw sensor windows.

**Architecture:**
```python
class CNNFeatureEncoder(nn.Module):
    def __init__(self, input_channels=6, hidden_dim=128):
        # Depthwise separable convolutions for efficiency
        self.conv1 = DepthwiseSeparableConv(input_channels, 64, kernel_size=5)
        self.conv2 = DepthwiseSeparableConv(64, 128, kernel_size=3)
        self.conv3 = DepthwiseSeparableConv(128, 256, kernel_size=3)
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x):
        # x: [batch, channels, time]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return self.gap(x).squeeze(-1)  # [batch, 256]
```

**Innovation:** Depthwise separable convolutions reduce parameters by 8-10x compared to standard convolutions.

### 6.3 Component 2: Transformer Temporal Module

**Purpose:** Model long-range temporal dependencies for boundary detection.

**Architecture:**
```python
class TemporalTransformer(nn.Module):
    def __init__(self, d_model=256, n_heads=4, n_layers=3):
        self.pos_encoding = LearnablePositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            EfficientTransformerLayer(d_model, n_heads)
            for _ in range(n_layers)
        ])
        
    def forward(self, x):
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x)
        return x
```

**Innovation:** Linear attention mechanism with O(n) complexity instead of O(n²).

### 6.4 Component 3: Self-Supervised Learning Module

This is the **core novelty** of our approach.

#### Pretext Task 1: Temporal Contrastive Learning

**Idea:** Segments from the same activity should have similar representations; segments from different activities should be dissimilar.

```python
def temporal_contrastive_loss(z_i, z_j, temperature=0.1):
    """
    Contrastive loss between temporal segments
    
    Args:
        z_i: Features from segment i [batch, dim]
        z_j: Features from segment j [batch, dim]
        temperature: Temperature parameter
    """
    # Positive pairs: adjacent segments from same activity
    # Negative pairs: segments from different activities or far apart
    
    # Compute similarity
    sim = F.cosine_similarity(z_i, z_j) / temperature
    
    # InfoNCE loss
    loss = -torch.log(
        torch.exp(sim) / 
        (torch.exp(sim) + sum_negative_exp)
    )
    
    return loss.mean()
```

#### Pretext Task 2: Continuity Prediction

**Idea:** Predict whether two segments belong to the same activity or represent a boundary.

```python
class ContinuityPredictor(nn.Module):
    def __init__(self, feature_dim=256):
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # Binary: same activity or boundary
        )
        
    def forward(self, z_i, z_j):
        # Concatenate features from two segments
        combined = torch.cat([z_i, z_j], dim=-1)
        return self.mlp(combined)
```

#### Pretext Task 3: Masked Temporal Modeling

**Idea:** Mask random time steps and predict them from context.

```python
def masked_temporal_loss(sensor_window, mask_ratio=0.15):
    """
    Mask random timesteps and predict them
    
    Similar to BERT's masked language modeling but for sensors
    """
    # Mask 15% of timesteps
    masked_input, mask = create_mask(sensor_window, mask_ratio)
    
    # Encode visible portions
    encoded = encoder(masked_input)
    
    # Decode to reconstruct
    reconstructed = decoder(encoded)
    
    # Loss only on masked positions
    loss = F.mse_loss(reconstructed[mask], sensor_window[mask])
    
    return loss
```

### 6.5 Component 4: Boundary Detection Module

**Purpose:** Identify activity transitions using attention scores.

```python
class BoundaryDetector(nn.Module):
    def __init__(self, d_model=256):
        self.attention = nn.MultiheadAttention(d_model, num_heads=4)
        self.boundary_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, temporal_features):
        # Self-attention captures dependencies
        attn_output, attn_weights = self.attention(
            temporal_features, 
            temporal_features, 
            temporal_features
        )
        
        # Boundary probability at each timestep
        boundary_probs = self.boundary_head(attn_output)
        
        return boundary_probs, attn_weights
```

### 6.6 Component 5: Edge Deployment Optimization

#### Knowledge Distillation

```python
class DistillationLoss(nn.Module):
    def __init__(self, temperature=3.0, alpha=0.7):
        self.temperature = temperature
        self.alpha = alpha
        
    def forward(self, student_logits, teacher_logits, labels):
        # Soft target loss
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=1),
            F.softmax(teacher_logits / self.temperature, dim=1),
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # Hard target loss
        hard_loss = F.cross_entropy(student_logits, labels)
        
        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss
```

#### Quantization-Aware Training

```python
class QuantizationAwareWrapper(nn.Module):
    def __init__(self, model):
        self.model = model
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        
    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x
```

---

## 7. Novel Contributions

### 7.1 Contribution 1: Temporal Contrastive Boundary Learning (TCBL)

**Novelty Score: 9.5/10**

**What:** A novel self-supervised pretext task specifically designed for temporal segmentation.

**Why Novel:** Existing SSL methods focus on classification; TCBL is the first to explicitly learn boundaries.

**Expected Impact:** Enable training with 10% of labels while maintaining >90% of supervised performance.

### 7.2 Contribution 2: Attention-Based Semantic Segmentation (ABSS)

**Novelty Score: 8.5/10**

**What:** Transformer-based boundary detection using semantic attention.

**Why Novel:** Extends Deep Similarity Segmentation with attention mechanisms for better long-range modeling.

**Expected Impact:** +5-7% improvement in boundary F1 over statistical similarity methods.

### 7.3 Contribution 3: Joint Segmentation-Classification Optimization (JSCO)

**Novelty Score: 8.0/10**

**What:** End-to-end optimization of segmentation and classification for edge deployment.

**Why Novel:** Existing methods optimize these separately; joint optimization improves efficiency.

**Expected Impact:** 30% reduction in parameters with <1% accuracy loss.

### 7.4 Contribution 4: Transitional Activity Specialization Module (TASM)

**Novelty Score: 8.5/10**

**What:** Specialized attention module for transitional activity detection.

**Why Novel:** First explicit focus on transitional activities in segmentation context.

**Expected Impact:** +8-12% improvement in transitional activity F1.

### 7.5 Overall Framework Novelty

**SAS-HAR Framework Novelty: 9.0/10**

The integration of all four contributions into a unified framework that:
- Learns boundaries without labels (TCBL)
- Models long-range dependencies (ABSS)
- Optimizes for edge deployment (JSCO)
- Excels at transitional activities (TASM)

This represents a **significant advancement** over existing HAR methods.

---

## 8. Experimental Plan

### 8.1 Datasets

| Dataset | Activities | Subjects | Duration | Modalities |
|---------|-----------|----------|----------|------------|
| **WISDM** | 6 | 36 | 54 hours | Acc |
| **UCI-HAR** | 6 | 30 | 8 hours | Acc, Gyro |
| **PAMAP2** | 18 | 9 | 10 hours | Acc, Gyro, Mag, HR |
| **Opportunity** | 5 | 4 | 6 hours | Multiple |

### 8.2 Evaluation Metrics

#### Segmentation Metrics
- **Boundary F1 Score:** Precision/recall of detected boundaries
- **Segmentation Accuracy:** Percentage of correctly classified segments
- **Hausdorff Distance:** Maximum boundary localization error

#### Classification Metrics
- **Overall Accuracy:** Standard classification accuracy
- **Weighted F1 Score:** Class-balanced F1
- **Transitional F1:** F1 specifically for transitional activities

#### Efficiency Metrics
- **Parameters:** Model size in thousands
- **FLOPs:** Floating point operations per inference
- **Memory:** RAM requirement during inference
- **Energy:** Joules per inference
- **Latency:** Milliseconds per inference

### 8.3 Baseline Methods

| Method | Category | Year | Key Feature |
|--------|----------|------|-------------|
| Fixed Sliding Window | Traditional | - | Standard baseline |
| Adaptive Sliding Window | Adaptive | 2017 | Energy-based sizing |
| Statistical Similarity | Similarity | 2023 | Hand-crafted features |
| Deep Similarity Segmentation | Similarity | 2024 | CNN-based |
| P2LHAP | Transformer | 2024 | Patch-based |
| TinierHAR | Lightweight | 2025 | 34K parameters |
| XTinyHAR | Distillation | 2025 | Cross-modal KD |

### 8.4 Experimental Protocol

#### Experiment 1: Self-Supervised Learning Evaluation
- **Goal:** Evaluate label efficiency of TCBL
- **Protocol:** Train with 1%, 5%, 10%, 50%, 100% of labels
- **Expected Result:** 90% of supervised performance with 10% labels

#### Experiment 2: Segmentation Accuracy
- **Goal:** Compare boundary detection methods
- **Protocol:** Evaluate on all datasets with full supervision
- **Expected Result:** >97% boundary F1, +5% over Deep Similarity

#### Experiment 3: Edge Deployment
- **Goal:** Measure efficiency metrics on real hardware
- **Protocol:** Deploy on Arduino Nano, STM32, measure energy
- **Expected Result:** <45 nJ/sample, <1 ms latency

#### Experiment 4: Transitional Activities
- **Goal:** Evaluate TASM module
- **Protocol:** Isolate transitional activities, compute F1
- **Expected Result:** >94% transitional F1, +8% over baseline

#### Experiment 5: Real-World Deployment
- **Goal:** Validate in unconstrained environment
- **Protocol:** Deploy on smartwatches, 10 users, 2 weeks each
- **Expected Result:** <5% accuracy degradation vs. lab

### 8.5 Ablation Studies

| Component | Ablation | Expected Impact |
|-----------|----------|-----------------|
| TCBL | Remove contrastive loss | -8% boundary F1 |
| TCBL | Remove continuity prediction | -5% boundary F1 |
| ABSS | Replace with similarity | -5% boundary F1 |
| JSCO | Optimize separately | +25% parameters |
| TASM | Remove transition attention | -10% transitional F1 |

---

## 9. Expected Impact

### 9.1 Scientific Impact

1. **New SSL Paradigm for Temporal Data:** TCBL establishes a new approach for learning temporal boundaries without labels, applicable beyond HAR to video segmentation, audio processing, and medical time series.

2. **Bridging Accuracy-Efficiency Gap:** Demonstrates that >98% accuracy and nanojoule-level efficiency are simultaneously achievable.

3. **Transitional Activity Understanding:** Advances fundamental understanding of how to model brief, dynamic human movements.

### 9.2 Practical Impact

1. **Healthcare Monitoring:** Enables 24/7 activity monitoring for elderly and disabled populations without battery concerns.

2. **Privacy Preservation:** All processing on-device eliminates privacy concerns of cloud-based systems.

3. **Accessibility:** Efficient models enable HAR on low-cost devices, democratizing access to activity monitoring.

### 9.3 Societal Impact

1. **Aging Population Support:** Continuous monitoring can detect early signs of mobility decline, enabling timely intervention.

2. **Fall Prevention:** Improved transitional activity detection can predict and prevent falls before they occur.

3. **Rehabilitation:** Precise activity tracking enables better monitoring of post-surgery recovery.

---

## 10. Research Timeline

### Year 1: Foundation (Months 1-12)

**Semester 1 (Months 1-6)**
- Month 1-2: Comprehensive literature review
- Month 3: Dataset preparation and preprocessing
- Month 4-5: Baseline reproduction (Deep Similarity Segmentation)
- Month 6: Year 1 report and proposal defense

**Semester 2 (Months 7-12)**
- Month 7-8: Implement SOTA baselines (P2LHAP, TinierHAR)
- Month 9-10: Design self-supervised learning framework
- Month 11-12: Initial TCBL experiments

**Milestones:**
- ✅ Proposal defended
- ✅ Baselines reproduced
- ✅ Initial TCBL results

### Year 2: Core Algorithm Development (Months 13-24)

**Semester 3 (Months 13-18)**
- Month 13-15: Implement attention-based segmentation
- Month 16-18: Design hybrid CNN-Transformer architecture

**Semester 4 (Months 19-24)**
- Month 19-21: Knowledge distillation experiments
- Month 22-24: Edge deployment on Arduino/STM32

**Milestones:**
- ✅ Working SAS-HAR prototype
- ✅ Paper 1 draft complete

### Year 3: Validation and Optimization (Months 25-36)

**Semester 5 (Months 25-30)**
- Month 25-27: Comprehensive benchmark evaluation
- Month 28-30: Ablation studies

**Semester 6 (Months 31-36)**
- Month 31-33: Real-world deployment study (10 users)
- Month 34-36: Paper 2 and Paper 3 drafts

**Milestones:**
- ✅ All experiments complete
- ✅ Paper 1 submitted
- ✅ Papers 2-3 drafted

### Year 4: Thesis and Dissemination (Months 37-48)

**Semester 7 (Months 37-42)**
- Month 37-39: Paper submissions and revisions
- Month 40-42: Thesis writing (chapters 1-4)

**Semester 8 (Months 43-48)**
- Month 43-45: Thesis writing (chapters 5-7)
- Month 46: Thesis submission
- Month 47: Defense preparation
- Month 48: PhD defense

**Milestones:**
- ✅ 3 papers published/accepted
- ✅ Thesis defended
- ✅ Open-source code released

---

## 11. Expected Publications

### Paper 1: Self-Supervised Temporal Segmentation for HAR

**Title:** "Temporal Contrastive Boundary Learning: A Self-Supervised Approach to Activity Segmentation"

**Target Venue:** IEEE Transactions on Biomedical Engineering (TBME)  
**Impact Factor:** 7.0 | **Acceptance Rate:** ~20%

**Abstract Preview:**
We present Temporal Contrastive Boundary Learning (TCBL), a novel self-supervised approach for discovering activity boundaries in continuous sensor streams without labeled segmentation data...

**Expected Submission:** Q2 2026  
**Expected Acceptance:** Q4 2026

---

### Paper 2: Attention-Based HAR Architecture

**Title:** "SAS-HAR: Self-Supervised Attention-based Segmentation for Human Activity Recognition"

**Target Venue:** NeurIPS 2026 or ICML 2026  
**Acceptance Rate:** ~25%

**Abstract Preview:**
We introduce SAS-HAR, a unified framework combining self-supervised boundary learning with attention-based temporal modeling for efficient activity recognition...

**Expected Submission:** Q2 2026  
**Expected Acceptance:** Q3 2026

---

### Paper 3: Edge AI Deployment

**Title:** "NanoHAR: Nanojoule-Level Human Activity Recognition with Self-Supervised Segmentation"

**Target Venue:** MLSys 2027 or tinyML Research Symposium  
**Acceptance Rate:** ~30%

**Abstract Preview:**
We present NanoHAR, the first HAR framework achieving nanojoule-level energy consumption while maintaining >98% accuracy through joint optimization of segmentation and classification...

**Expected Submission:** Q4 2026  
**Expected Acceptance:** Q1 2027

---

## 12. References

[1] Bulling, A., Blanke, U., & Schiele, B. (2014). A tutorial on human activity recognition using body-worn inertial sensors. ACM Computing Surveys, 46(3), 1-33.

[2] Shiri, F. M., Perumal, T., Mustapha, N., & Mohamed, R. (2025). Deep learning and federated learning in human activity recognition with sensor data: A comprehensive review. Computer Modeling in Engineering & Sciences, 145(2), 1389-1485.

[3] United Nations. (2019). World Population Prospects 2019: Highlights. New York: United Nations.

[4] Alohali, M. A., et al. (2025). Advanced smart human activity recognition system for disabled people using artificial intelligence. Scientific Reports, 15(1), 31372.

[5] Shahiduzzaman, K. (2025). UActivity: A user-specific human activity recognition and fall detection for elderly care. Engineering, Technology & Applied Science Research, 15(6), 30169-74.

[6] Ismael, M. N. (2025). Comparative analysis of machine learning methods for human activity recognition using wearable sensors. Scientia, 2(11), 114-31.

[7] Baraka, A. M. A., & Mohd Noor, M. H. (2023). Similarity segmentation approach for sensor-based activity recognition. IEEE Sensors Journal, 23(17), 19704-16.

[8] Noor, M. H. M., et al. (2017). Adaptive sliding window segmentation for physical activity recognition. (Conference paper).

[9] Perera, C. K., et al. (2024). A motion capture dataset on human sitting to walking transitions. Scientific Data, 11, 878.

[10] Wang, X., et al. (2025). A dual-task improved transformer framework for decoding lower limb sit-to-stand movement. Machines, 13(10), 953.

[11] Vaswani, A., et al. (2017). Attention is all you need. NeurIPS.

[12] Li, S., et al. (2024). P2LHAP: Wearable sensor-based HAR, segmentation and forecast through patch-to-label Seq2Seq Transformer. arXiv:2403.08214.

[13] Lamaakal, I., et al. (2025). XTinyHAR: A tiny inertial transformer for HAR via multimodal knowledge distillation. Scientific Reports, Nature.

[14] Miao, M., et al. (2025). Attention-based CNN-BiGRU-Transformer model for HAR. Applied Sciences, 15(23).

[15] Noor, M. H. M., et al. (2017). Adaptive sliding window segmentation for physical activity recognition.

[16] Baraka, A. M. A., & Mohd Noor, M. H. (2023). Similarity segmentation approach for sensor-based activity recognition. IEEE Sensors Journal.

[17] Baraka, A., & Mohd Noor, M. H. (2024). Deep similarity segmentation model for sensor-based activity recognition. Multimedia Tools and Applications.

[18] Chen, T., et al. (2020). A simple framework for contrastive learning of visual representations. ICML.

[19] Devlin, J., et al. (2019). BERT: Pre-training of deep bidirectional transformers. NAACL.

[20] Dosovitskiy, A., et al. (2021). An image is worth 16x16 words: Transformers for image recognition at scale. ICLR.

[21] Brown, T., et al. (2020). Language models are few-shot learners. NeurIPS.

[22] Haque, S. T., et al. (2024). LightHART: Lightweight human activity recognition transformer. Pattern Recognition.

[23] Bian, S., et al. (2025). TinierHAR: Towards ultra-lightweight deep learning models. arXiv:2507.07949.

[24] Li, S., et al. (2024). P2LHAP. arXiv:2403.08214.

[25] Bacellar, A. T. L., et al. (2025). nanoML for human activity recognition. tinyML Research Symposium.

[26] Bian, S., et al. (2025). TinierHAR. UbiComp/ISWC '25.

---

*Document Version: 1.0*  
*Last Updated: March 2026*  
*Prepared by: Mohammed Jasim*
