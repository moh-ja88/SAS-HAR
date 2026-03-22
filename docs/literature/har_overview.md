# Literature Review: Human Activity Recognition (HAR) Overview

## 1. Introduction

Human Activity Recognition (HAR) is the computational task of identifying and classifying human activities from sensor data. This document provides a comprehensive overview of the HAR field, its applications, challenges, and evolution.

---

## 2. HAR Pipeline

### Standard HAR Pipeline

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│   Sensor    │────▶│  Preprocessing│────▶│ Segmentation│────▶│ Feature      │
│   Data      │     │  & Filtering  │     │ (Windowing) │     │ Extraction   │
└─────────────┘     └──────────────┘     └─────────────┘     └──────────────┘
                                                                   │
                                                                   ▼
                                                            ┌──────────────┐
                                                            │Classification│
                                                            │   Model      │
                                                            └──────────────┘
                                                                   │
                                                                   ▼
                                                            ┌──────────────┐
                                                            │   Activity   │
                                                            │   Labels     │
                                                            └──────────────┘
```

### Pipeline Components

| Component | Purpose | Common Approaches |
|-----------|---------|-------------------|
| **Sensor Data** | Raw measurements | Accelerometer, Gyroscope, Magnetometer |
| **Preprocessing** | Noise reduction | Filtering, normalization, interpolation |
| **Segmentation** | Temporal division | Fixed sliding window, adaptive window |
| **Feature Extraction** | Representation learning | Hand-crafted, CNN, LSTM, Transformer |
| **Classification** | Activity labeling | SVM, RF, CNN, LSTM, Transformer |

---

## 3. Sensor Modalities

### 3.1 Inertial Measurement Units (IMUs)

**Accelerometer**
- Measures linear acceleration (m/s²)
- 3-axis: x, y, z
- Sampling rate: 20-100 Hz typical
- Applications: Movement detection, step counting

**Gyroscope**
- Measures angular velocity (°/s)
- 3-axis: roll, pitch, yaw
- Sampling rate: 20-100 Hz typical
- Applications: Rotation detection, orientation

**Magnetometer**
- Measures magnetic field (μT)
- 3-axis: x, y, z
- Applications: Heading estimation, orientation

### 3.2 Other Modalities

| Modality | Advantages | Disadvantages |
|----------|------------|---------------|
| **Video** | Rich information, interpretable | Privacy concerns, high compute |
| **Audio** | Contextual information | Privacy, background noise |
| **Heart Rate** | Physiological state | Requires dedicated sensor |
| **GPS** | Location context | Outdoor only, privacy |
| **WiFi/Bluetooth** | Environment sensing | Indirect, variable quality |

### 3.3 Sensor Placement

| Location | Activities Best Detected | Comfort Level |
|----------|-------------------------|---------------|
| **Wrist** | Daily activities, gestures | High (watch) |
| **Waist** | Full-body movements | Medium |
| **Chest** | Heart rate, breathing | Low |
| **Ankle** | Walking, running | Medium |
| **Pocket** | Daily activities | High |

---

## 4. Activity Taxonomy

### 4.1 Activity Categories

```
Human Activities
├── Static Activities
│   ├── Sitting
│   ├── Standing
│   ├── Lying down
│   └── Static postures
│
├── Dynamic Activities
│   ├── Walking
│   ├── Running
│   ├── Cycling
│   ├── Stairs (up/down)
│   └── Jumping
│
├── Transitional Activities
│   ├── Sit-to-Stand
│   ├── Stand-to-Sit
│   ├── Sit-to-Lie
│   ├── Lie-to-Sit
│   └── Stand-to-Walk
│
├── Complex Activities
│   ├── Cooking
│   ├── Cleaning
│   ├── Eating
│   └── Working
│
└── Falls (Critical)
    ├── Forward fall
    ├── Backward fall
    ├── Lateral fall
    └── Near-fall (stumble)
```

### 4.2 Activity Characteristics

| Type | Duration | Dynamics | F1 Range | Challenge |
|------|----------|----------|----------|-----------|
| Static | Extended | Low | 95-99% | Distinguishing similar postures |
| Dynamic | Variable | High | 92-97% | Duration variability |
| **Transitional** | **1-3s** | **Very High** | **70-85%** | **Short + Variable + Dynamic** |
| Complex | Variable | Medium | 80-92% | Multi-component, ambiguous |
| Falls | <1s | Extreme | 90-95% | Rarity, critical importance |

---

## 5. Applications

### 5.1 Healthcare

| Application | Activities Monitored | Impact |
|-------------|---------------------|--------|
| **Ambient Assisted Living (AAL)** | Daily activities, transitions | Enable elderly independence |
| **Fall Detection** | Falls, near-falls | Emergency response |
| **Rehabilitation** | Exercise movements | Track recovery progress |
| **Parkinson's Monitoring** | Gait, tremors | Early symptom detection |
| **Sleep Monitoring** | Sleep postures, movements | Sleep quality assessment |

### 5.2 Sports and Fitness

| Application | Activities | Value |
|-------------|-----------|-------|
| **Activity Tracking** | Steps, exercise | Health motivation |
| **Sports Analytics** | Sport-specific movements | Performance improvement |
| **Form Correction** | Exercise form | Injury prevention |

### 5.3 Smart Environments

| Application | Activities | Technology |
|-------------|-----------|------------|
| **Smart Home** | ADLs, presence | IoT integration |
| **Workplace Safety** | Hazardous movements | Industrial monitoring |
| **Retail Analytics** | Customer behavior | Business intelligence |

---

## 6. Evolution of HAR Methods

### Timeline

```
2000s: Traditional ML Era
├── Hand-crafted features (mean, variance, FFT)
├── Classifiers: SVM, Random Forest, HMM
└── Accuracy: 80-90%

2010-2015: Deep Learning Emergence
├── CNNs for automatic feature learning
├── LSTMs for temporal modeling
└── Accuracy: 90-95%

2015-2020: Attention and Hybrid Models
├── Attention mechanisms
├── CNN-LSTM hybrids
├── Transfer learning
└── Accuracy: 93-97%

2020-2024: Transformers and Foundation Models
├── Vision Transformers for HAR
├── Self-supervised learning
├── Multimodal fusion
└── Accuracy: 95-99%

2024-2026: Edge AI and Specialization
├── TinyML deployment
├── Specialized architectures (<25K params)
├── Foundation models for HAR
├── Self-supervised segmentation
└── Accuracy: 97-99%, Energy: <100 nJ
```

### Method Comparison

| Era | Method | Features | Accuracy | Limitation |
|-----|--------|----------|----------|------------|
| **Traditional** | SVM + Stats | Hand-crafted | 80-90% | Feature engineering |
| **Early DL** | CNN | Learned | 90-93% | Limited temporal |
| **Hybrid** | CNN-LSTM | Learned | 93-96% | Sequential processing |
| **Attention** | CNN-Attention | Learned | 95-97% | Computational cost |
| **Transformer** | ViT, Swin | Learned | 96-99% | Data hungry, large |
| **Edge AI** | Lightweight | Learned | 95-98% | Accuracy-efficiency tradeoff |

---

## 7. Datasets

### Major Public Datasets

| Dataset | Year | Activities | Subjects | Duration | Sensors |
|---------|------|------------|----------|----------|---------|
| **UCI-HAR** | 2012 | 6 | 30 | 8 hours | Acc, Gyro |
| **WISDM** | 2012 | 6 | 36 | 54 hours | Acc |
| **PAMAP2** | 2013 | 18 | 9 | 10 hours | Acc, Gyro, Mag, HR |
| **Opportunity** | 2010 | 5 | 4 | 6 hours | Multiple |
| **REALDISP** | 2013 | 33 | 17 | - | IMU |
| **Daphnet** | 2013 | 3 | 10 | 6 hours | Acc |
| **ExtraSensory** | 2016 | 51 | 60 | - | Phone sensors |
| **MobiAct** | 2016 | 15 | 67 | - | Acc, Gyro |
| **RealWorld** | 2018 | 8 | 15 | - | Acc, Gyro, Mag, GPS |

### Dataset Selection Criteria

For our research, we select datasets based on:
1. **Public availability**: Open access for reproducibility
2. **Multiple subjects**: For cross-subject validation
3. **Transitional activities**: For our specialized focus
4. **Sensor diversity**: Acc + Gyro minimum
5. **Established benchmarks**: For fair comparison

**Selected: WISDM, UCI-HAR, PAMAP2, Opportunity**

---

## 8. Evaluation Metrics

### Classification Metrics

| Metric | Formula | Use Case |
|--------|---------|----------|
| **Accuracy** | (TP + TN) / Total | Overall performance |
| **Precision** | TP / (TP + FP) | Confidence in predictions |
| **Recall** | TP / (TP + FN) | Coverage of actual positives |
| **F1-Score** | 2 * (P * R) / (P + R) | Balanced P and R |
| **Weighted F1** | Σ (F1_i * n_i) / N | Imbalanced classes |

### Segmentation Metrics

| Metric | Description | Use Case |
|--------|-------------|----------|
| **Boundary F1** | Precision/recall of boundaries | Segmentation accuracy |
| **Segmentation Accuracy** | Correctly classified segments | Segment-level performance |
| **Hausdorff Distance** | Max boundary localization error | Boundary precision |
| **Rand Index** | Agreement between segmentations | Segment similarity |

### Efficiency Metrics

| Metric | Unit | Target (Edge) |
|--------|------|---------------|
| **Parameters** | Count | <50K |
| **Model Size** | KB/MB | <500 KB |
| **FLOPs** | Operations | <10M |
| **Inference Latency** | ms | <10 ms |
| **Energy** | nJ/sample | <100 nJ |
| **Memory** | KB | <100 KB |

---

## 9. Current Challenges

### 9.1 Technical Challenges

1. **Segmentation Accuracy**
   - Fixed windows cut across boundaries
   - Adaptive methods have limited accuracy
   - Need for semantic segmentation

2. **Transitional Activities**
   - Short duration (1-3s)
   - High variability
   - Poor performance (70-85% F1)

3. **Real-World Deployment**
   - Lab datasets vs. real conditions
   - 5-10% accuracy degradation
   - User variability

4. **Energy Constraints**
   - Battery life limitations
   - Need for nanojoule-level efficiency
   - Accuracy-efficiency tradeoff

### 9.2 Data Challenges

1. **Annotation Cost**
   - Expert annotation: $50-100/hour
   - Boundary annotation: Subjective
   - Limited labeled data

2. **Dataset Imbalance**
   - Static activities dominate
   - Transitions underrepresented
   - Falls extremely rare

3. **Domain Shift**
   - Lab to real-world
   - Device to device
   - User to user

### 9.3 Practical Challenges

1. **Privacy Concerns**
   - Continuous monitoring
   - Sensitive data
   - Cloud processing risks

2. **User Acceptance**
   - Comfort of wearables
   - Battery charging
   - False alarms

3. **Regulatory Compliance**
   - Medical device regulations
   - Data protection (GDPR, HIPAA)
   - Clinical validation requirements

---

## 10. Future Directions

### Emerging Trends

1. **Self-Supervised Learning**
   - Reduce annotation cost
   - Learn from unlabeled data
   - Foundation models for HAR

2. **Edge AI / TinyML**
   - On-device processing
   - Nanojoule-level efficiency
   - Privacy-preserving

3. **Multimodal Fusion**
   - IMU + video + audio
   - Robust to missing modalities
   - Richer representations

4. **Personalization**
   - User-specific adaptation
   - Continual learning
   - Federated learning

5. **Explainable AI**
   - Interpretable predictions
   - Clinical trust
   - Debug and improve

---

## 11. References

Key foundational papers:

1. Bulling, A., Blanke, U., & Schiele, B. (2014). A tutorial on human activity recognition using body-worn inertial sensors. ACM Computing Surveys.

2. Wang, J., et al. (2019). Deep learning for sensor-based activity recognition: A survey. Pattern Recognition Letters.

3. Shiri, F. M., et al. (2025). Deep learning and federated learning in human activity recognition. Computer Modeling in Engineering & Sciences.

4. Recent surveys (2023-2026) on HAR, edge AI, and self-supervised learning.

---

*Last Updated: March 2026*
