# Self-Supervised Learning for HAR — Literature Review (2023–2026)

**Scope:** Self-Supervised Learning (SSL) for Human Activity Recognition (HAR) with wearable IMU sensors. Three SSL paradigms covered: (A) Contrastive Learning, (B) Predictive/Generative Pretext, (C) JEPA. Special attention to boundary detection, temporal segmentation, and adaptive windowing.

**Sources:** arXiv API (300+ papers retrieved, 40+ curated below), Semantic Scholar. All papers from 2023 onwards.

---

## A. Contrastive SSL for HAR

### A1. Timestamp-supervised Wearable-based Activity Segmentation and Recognition with Contrastive Learning and Order-Preserving Optimal Transport
- **Authors:** Songpengcheng Xia, Lei Chu, Ling Pei, Jiarui Yang, Wenxian Yu, Robert C. Qiu
- **Year:** 2023 | **Venue:** arXiv (2310.09114)
- **Key Contribution:** Proposes joint activity segmentation + recognition using only timestamp-level supervision (one label per activity segment). Uses contrastive sample-to-prototype learning and optimal transport for pseudo-label generation to bridge the gap between sparse annotations and dense segmentation.
- **Boundary Detection Relevance:** **DIRECTLY ADDRESSES** boundary detection. Tackles the sliding-window multi-class window problem by jointly segmenting and recognizing activities. Demonstrates that contrastive learning + optimal transport can effectively locate activity boundaries with minimal supervision.

### A2. TS-MoCo: Time-Series Momentum Contrast for Self-Supervised Physiological Representation Learning
- **Authors:** Philipp Hallgarten, David Bethage, Ozan Özdenizci, Tobias Grosse-Puppendahl, Enkelejda Kasneci
- **Year:** 2023 | **Venue:** arXiv (2306.06522)
- **Key Contribution:** Adapts MoCo (momentum contrast) for multivariate time-series physiological data. Uses a transformer encoder pretrained via contrastive momentum learning. Evaluated on HAR from inertial sensors and emotion recognition from EEG.
- **Boundary Detection Relevance:** Indirectly relevant — learns window-level representations that could be used for boundary-sensitive downstream tasks, but does not explicitly address segmentation.

### A3. Multimodal Contrastive Learning with Hard Negative Sampling for Human Activity Recognition
- **Authors:** Hyeongju Choi, Apoorva Beedu, Irfan Essa
- **Year:** 2023 | **Venue:** arXiv (2309.01262)
- **Key Contribution:** Introduces hard negative sampling for multimodal HAR (skeleton + IMU pairs). Uses an adjustable concentration parameter to exploit hard negatives that are semantically different but close in latent space. Outperforms prior SOTA on UTD-MHAD.
- **Boundary Detection Relevance:** Indirectly relevant — hard negative mining could help distinguish activity transition boundaries (where similar-but-different activities co-occur).

### A4. Contrastive Left-Right Wearable Sensors (IMUs) Consistency Matching for HAR
- **Authors:** Dominique Nshimyimana, Vitor Fortes Rey, Paul Lukowicz
- **Year:** 2023 | **Venue:** arXiv (2311.12674)
- **Key Contribution:** Exploits body symmetry as a natural positive pair for contrastive learning — co-occurring left/right IMU sensor data. No data augmentation needed. Significant improvement over SimCLR on Opportunity and MM-Fit datasets.
- **Boundary Detection Relevance:** Indirectly relevant — the temporal co-occurrence constraint (left/right sensors during same activity) could be extended to detect activity transitions via co-occurrence breakdown.

### A5. CroSSL: Cross-modal Self-Supervised Learning for Time-series through Latent Masking
- **Authors:** Shohreh Deldari, Dimitris Spathis, Mohammad Malekzadeh, Fahim Kawsar, Flora Salim, Akhil Mathur
- **Year:** 2023 | **Venue:** arXiv (2307.16847)
- **Key Contribution:** Novel cross-modal SSL that masks intermediate modality embeddings (not raw data) and aggregates them via a cross-modal aggregator. Handles missing modalities without negative-pair sampling. Evaluated on motion sensors (accelerometer, gyroscope) and biosignals.
- **Boundary Detection Relevance:** Partially relevant — latent masking of modalities could be adapted to detect temporal boundaries by masking temporal segments rather than modalities.

### A6. Don't Freeze: Finetune Encoders for Better Self-Supervised HAR
- **Authors:** Vitor Fortes Rey, Dominique Nshimyimana, Paul Lukowicz
- **Year:** 2023 | **Venue:** arXiv (2307.01168)
- **Key Contribution:** Shows that the standard "pretrain-freeze-finetune" paradigm is suboptimal for HAR. Simply not freezing the encoder during finetuning yields substantial gains across multiple pretext tasks (reconstruction, contrastive predictive coding) and datasets.
- **Boundary Detection Relevance:** Indirectly relevant — provides practical guidance for deploying SSL encoders that could be used in boundary-sensitive pipelines.

### A7. Diffusion Model-based Contrastive Learning for Human Activity Recognition (CLAR)
- **Authors:** Chunjing Xiao, Yanhui Han, Wei Yang, Yane Hou, Fangzhan Shi, Kevin Chetty
- **Year:** 2024 | **Venue:** arXiv (2408.05567)
- **Key Contribution:** Uses a diffusion model to generate high-quality augmented data for contrastive learning on WiFi CSI. Separates high/low frequency conditioning to reduce distortion. Introduces adaptive sample weighting.
- **Boundary Detection Relevance:** Indirectly relevant — diffusion-based augmentation could help model activity transitions (boundaries) by generating synthetic transition samples.

### A8. Guidelines for Augmentation Selection in Contrastive Learning for Time Series Classification
- **Authors:** Ziyu Liu, Azadeh Alavi, Minyi Li, Xiang Zhang
- **Year:** 2024 | **Venue:** arXiv (2407.09336)
- **Key Contribution:** Establishes a principled framework for selecting augmentations based on time-series characteristics (trend, seasonality). Evaluates 8 augmentations on 12 synthetic + 6 real-world datasets including HAR. Provides a recommendation algorithm achieving Recall@3 of 0.667.
- **Boundary Detection Relevance:** Indirectly relevant — augmentation choice directly impacts how well contrastive representations capture transition/boundary dynamics.

### A9. RelCon: Relative Contrastive Learning for a Motion Foundation Model for Wearable Data
- **Authors:** Maxwell A. Xu, Jaya Narain, Gregory Darnell, Haraldur Hallgrimisson, Hyewon Jeong, Darren Forde, Richard Fineman, Karthik J. Raghuram, James M. Rehg, Shirley Ren
- **Year:** 2024 | **Venue:** arXiv (2411.18822)
- **Key Contribution:** First motion foundation model trained via relative contrastive learning. Learns a distance measure capturing motif similarity and rotation invariance. Trained on 1 billion segments from 87,376 participants. SOTA on HAR and gait metric regression.
- **Boundary Detection Relevance:** Indirectly relevant — relative distance learning between motion segments could be applied to detecting activity transitions/boundaries.

### A10. Subject Invariant Contrastive Learning for Human Activity Recognition (SICL)
- **Authors:** Yavuz Yarici, Kiran Kokilepersaud, Mohit Prabhushankar, Ghassan AlRegib
- **Year:** 2025 | **Venue:** arXiv (2507.03250)
- **Key Contribution:** Re-weights negative pairs from the same subject to suppress subject-specific variations and emphasize activity-specific features. Up to 11% improvement over standard contrastive learning on UTD-MHAD, MMAct, and DARai.
- **Boundary Detection Relevance:** Indirectly relevant — subject-invariant representations could improve boundary detection generalization across users.

### A11. Auto-Augmentation Contrastive Learning for Wearable-based HAR (AutoCL)
- **Authors:** Qingyu Wu, Jianfei Shen, Feiyi Fan, Yang Gu, Chenyang Xu, Yiqiang Chen
- **Year:** 2026 | **Venue:** arXiv (2602.02542)
- **Key Contribution:** End-to-end auto-augmentation for contrastive HAR using a Siamese network with an embedded generator that learns augmentations in latent space. Incorporates stop-gradient and correlation reduction strategies.
- **Boundary Detection Relevance:** Indirectly relevant — automated augmentation could learn to preserve/perturb boundary-relevant features.

### A12. BenchHAR: Benchmarking Self-Supervised Learning for Generalizable Sensor-based HAR
- **Authors:** Yize Cai, Rui Feng, Anlan Yu, Baoshen Guo, Zhiqing Hong
- **Year:** 2026 | **Venue:** arXiv (2605.08296)
- **Key Contribution:** Comprehensive benchmark evaluating 8 SSL methods across 12 encoder-classifier architectures on a curated 258K-sample dataset. Systematically compares generalization to unseen target distributions for sensor-based HAR.
- **Boundary Detection Relevance:** Provides essential baselines for evaluating boundary-detection methods in an SSL context.

### A13. Closing the Modality Gap in Zero-Shot HAR: Contrastive Training and Separability-Optimized Prototypes on IMU Data
- **Authors:** (See arXiv 2606.10789)
- **Year:** 2026 | **Venue:** arXiv (2606.10789)
- **Key Contribution:** Bridges sensor-semantic gap for zero-shot HAR via contrastive training with separability-optimized prototypes on IMU embeddings.
- **Boundary Detection Relevance:** Indirectly relevant — zero-shot capability is useful for detecting novel activity boundaries without retraining.

### A14. Contrastive Learning for Multimodal HAR with Limited Labeled Data
- **Authors:** (See arXiv 2604.23281)
- **Year:** 2026 | **Venue:** arXiv (2604.23281)
- **Key Contribution:** Contrastive learning for multi-source sensor fusion HAR under label scarcity, using collaborative sensing.
- **Boundary Detection Relevance:** Indirectly relevant — multimodal contrastive embeddings could improve cross-modal boundary consistency.

---

## B. Predictive/Generative SSL for HAR (Masked Modeling, Reconstruction)

### B1. An Improved Masking Strategy for Self-supervised Masked Reconstruction in HAR
- **Authors:** (See arXiv 2312.04147)
- **Year:** 2023 | **Venue:** arXiv (2312.04147)
- **Key Contribution:** Proposes improved masking strategies for masked autoencoder-style SSL specifically for HAR. Optimizes mask ratios and patterns for wearable sensor time-series.
- **Boundary Detection Relevance:** Indirectly relevant — masking strategies determine how well the model learns temporal dependencies across potential activity boundaries.

### B2. Frequency-Aware Masked Autoencoders for HAR using Accelerometers
- **Authors:** (See arXiv 2502.17477)
- **Year:** 2025 | **Venue:** arXiv (2502.17477)
- **Key Contribution:** Frequency-aware masked autoencoder that operates in both time and frequency domains for accelerometer-based HAR. Captures multi-scale temporal patterns through frequency-domain masking.
- **Boundary Detection Relevance:** Partially relevant — frequency-domain analysis is well-suited for detecting abrupt activity transitions/boundaries.

### B3. HiMAE: Hierarchical Masked Autoencoders Discover Resolution-Specific Structure in Wearable Time Series
- **Authors:** (See arXiv 2510.25785)
- **Year:** 2025 | **Venue:** arXiv (2510.25785)
- **Key Contribution:** Hierarchical masked autoencoder that discovers multi-resolution structure in wearable signals. Different masking levels capture patterns at different temporal granularities.
- **Boundary Detection Relevance:** **STRONGLY RELEVANT** — hierarchical/multi-resolution structure is directly applicable to detecting boundaries at different temporal scales (fine-grained transitions vs. macro activity changes).

### B4. MU-MAE: Multimodal Masked Autoencoders-Based One-Shot Learning
- **Authors:** (See arXiv 2408.04243)
- **Year:** 2024 | **Venue:** arXiv (2408.04243)
- **Key Contribution:** Multimodal masked autoencoder for one-shot learning across modalities. Pretrained via masked reconstruction of multiple sensor streams simultaneously.
- **Boundary Detection Relevance:** Indirectly relevant — multimodal reconstruction errors could signal activity transitions where modality relationships change.

### B5. PIM: Physics-Informed Multi-task Pre-training for Improving Inertial Sensor-Based HAR
- **Authors:** (See arXiv 2503.17978)
- **Year:** 2025 | **Venue:** arXiv (2503.17978)
- **Key Contribution:** Physics-informed multi-task pretraining incorporating IMU physical constraints (gravity, body kinematics) as inductive biases for SSL.
- **Boundary Detection Relevance:** Partially relevant — physics-informed features may capture transition dynamics (e.g., acceleration changes during activity switches) that define boundaries.

### B6. HAR-DoReMi: Optimizing Data Mixture for Self-Supervised HAR Across Heterogeneous IMU Datasets
- **Authors:** (See arXiv 2503.13542)
- **Year:** 2025 | **Venue:** arXiv (2503.13542)
- **Key Contribution:** Addresses data mixture optimization for SSL pretraining across heterogeneous HAR datasets. Dynamically reweights data domains during pretraining.
- **Boundary Detection Relevance:** Indirectly relevant — optimal data mixtures ensure the model sees sufficient transition/boundary examples.

### B7. Physical Self-Supervised Learning: IMU Sensing without Manual Labels
- **Authors:** (See arXiv 2607.18361)
- **Year:** 2026 | **Venue:** arXiv (2607.18361)
- **Key Contribution:** Physics-grounded SSL for IMU that uses physical laws as pretext tasks, eliminating manual label dependency. Addresses robustness to sensor heterogeneity.
- **Boundary Detection Relevance:** Partially relevant — physical pretext tasks (e.g., predicting sensor orientation changes) may naturally capture activity transition points.

### B8. A Foundation Model for Wearable Movement Data in Mental Health Research
- **Authors:** (See arXiv 2411.15240)
- **Year:** 2024 | **Venue:** arXiv (2411.15240)
- **Key Contribution:** Large-scale pretrained foundation model for wearable movement data using masked reconstruction. Applied to mental health biomarker discovery.
- **Boundary Detection Relevance:** Indirectly relevant — foundation model representations could be applied to activity segmentation in clinical settings.

### B9. Wearable Accelerometer Foundation Models for Health via Knowledge Distillation
- **Authors:** (See arXiv 2412.11276)
- **Year:** 2024 | **Venue:** arXiv (2412.11276)
- **Key Contribution:** Distills knowledge from large teacher models into efficient wearable accelerometer foundation models for health monitoring tasks.
- **Boundary Detection Relevance:** Indirectly relevant — distilled models enable real-time boundary detection on edge devices.

### B10. WavesFM: Hierarchical Representation Learning for Longitudinal Wearable Sensor Waveforms
- **Authors:** (See arXiv 2605.09173)
- **Year:** 2026 | **Venue:** arXiv (2605.09173)
- **Key Contribution:** Hierarchical SSL for longitudinal wearable waveforms (PPG, accelerometry). Captures patterns from sub-second to daily timescales.
- **Boundary Detection Relevance:** Partially relevant — hierarchical temporal representations naturally support multi-scale boundary detection.

### B11. Inertia-1: An Open Exploration of Wearable Motion Foundation Models
- **Authors:** (See arXiv 2607.06617)
- **Year:** 2026 | **Venue:** arXiv (2607.06617)
- **Key Contribution:** Comprehensive open exploration of wearable motion foundation models trained on 18.2M+ hours of accelerometer data. Studies the full lifecycle: data choices, model architectures, training objectives. Evaluates across 15 datasets.
- **Boundary Detection Relevance:** Indirectly relevant — provides scaling laws and design recipes critical for deploying SSL models in boundary-aware HAR systems.

### B12. SPAR: Self-supervised Placement-Aware Representation Learning for Distributed Sensing
- **Authors:** Yizhuo Chen, Tianchen Wang, You Lyu, Yanlan Hu, Jinyang Li, Tomoyoshi Kimura, Hongjue Zhao, Yigong Hu, Denizhan Kara, Tarek Abdelzaher
- **Year:** 2025 | **Venue:** arXiv (2505.16936)
- **Key Contribution:** Introduces placement-aware SSL that models signal-position duality through spatial/structural positional embeddings and dual reconstruction objectives. Explicitly treats sensor placement as intrinsic to representation learning.
- **Boundary Detection Relevance:** Partially relevant — placement-aware representations could help disentangle location-specific artifacts from genuine activity transitions.

### B13. Context-Aware Predictive Coding: A Representation Learning Framework for WiFi Sensing
- **Authors:** (See arXiv 2410.01825)
- **Year:** 2024 | **Venue:** arXiv (2410.01825)
- **Key Contribution:** Combines predictive coding with context-awareness for WiFi CSI-based HAR. Predictive coding pretext captures temporal dependencies.
- **Boundary Detection Relevance:** Partially relevant — predictive coding naturally models temporal predictability changes at activity boundaries.

### B14. Masked Video and Body-worn IMU Autoencoder for Egocentric Action Recognition
- **Authors:** (See arXiv 2407.06628)
- **Year:** 2024 | **Venue:** arXiv (2407.06628)
- **Key Contribution:** Joint masked autoencoder over video and body-worn IMU signals for egocentric action recognition. Cross-modal reconstruction.
- **Boundary Detection Relevance:** Partially relevant — joint multimodal masking could identify boundaries where video and IMU modalities diverge.

---

## C. JEPA (Joint-Embedding Predictive Architecture) for Time Series / HAR

### C1. HAR-JEPA: Joint-Embedding Predictive Architecture for Sensor-based Activity Recognition
- **Authors:** Mohd Halim Mohd Noor, Abdulrahman M. A. Baraka
- **Year:** 2026 | **Venue:** arXiv (2607.16350)
- **Key Contribution:** **First JEPA framework directly for sensor-based HAR.** Encoder models fine-grained local temporal patterns within windows AND long-term sequences across adjacent windows. Introduces improved VICReg objective with norm regularization to prevent collapse.
- **Boundary Detection Relevance:** **HIGHLY RELEVANT** — explicitly evaluates on continuous activity datasets. Shows superior generalization on transitional activities (sit-to-stand, sit-to-lie) where supervised models overfit. Transition activities ARE boundary events. The multi-window temporal modeling directly relates to boundary detection.

### C2. CF-JEPA: Mask-free Forward Prediction for Time-Series Representation Learning
- **Authors:** Jaehoon Lee, Sunghyun Sim
- **Year:** 2026 | **Venue:** arXiv (2606.07031)
- **Key Contribution:** Replaces masking with multi-horizon forward prediction (short, mid, long-horizon) using random crops as context views. Eliminates the temporal continuity disruption of masking. Discovered asymmetry: online encoder for classification, EMA target encoder for forecasting. SOTA on UCR/UEA benchmarks.
- **Boundary Detection Relevance:** **STRONGLY RELEVANT** — multi-horizon forward prediction naturally models temporal structure changes that define boundaries. The mask-free approach preserves temporal continuity critical for boundary detection.

### C3. HEPA: Self-Supervised Horizon-Conditioned Event Predictive Architecture for Time Series
- **Authors:** Jonas Petersen, Gian-Alessandro Lombardi, Riccardo Maggioni, Camilla Mazzoleni, Federico Martelli, Philipp Petersen
- **Year:** 2026 | **Venue:** arXiv (2605.11130)
- **Key Contribution:** JEPA variant for critical event prediction. Horizon-conditioned predictor forecasts future representations (not values) in latent space. Produces survival CDF over prediction horizons. Outperforms PatchTST, iTransformer, MAE, Chronos-2 across 14 benchmarks.
- **Boundary Detection Relevance:** **HIGHLY RELEVANT** — event prediction in latent space is directly analogous to boundary detection. The horizon-conditioned framework naturally captures "when will the next change occur."

### C4. TS-JEPA: Joint Embeddings Go Temporal
- **Authors:** Sofiane Ennadir, Siavash Golkar, Leopoldo Sarra
- **Year:** 2025 | **Venue:** arXiv (2509.25449)
- **Key Contribution:** First JEPA architecture specifically adapted for time-series representation learning. Validates on classification and forecasting, matching or surpassing SOTA baselines. Lays groundwork for time-series foundation models based on joint embedding.
- **Boundary Detection Relevance:** Partially relevant — temporal JEPA representations could be applied to boundary detection, though not explicitly evaluated for this.

### C5. SC-JEPA: Stabilizing Latent Predictive Learning for Time-Series Anomaly Prediction
- **Authors:** Yanan He, Yunshi Wen, Xin Wang, Tengfei Ma
- **Year:** 2026 | **Venue:** arXiv (2602.04643)
- **Key Contribution:** Addresses JEPA instability for time-series via soft codebook bottleneck for discretized predictive state space. Multi-resolution predictive objective captures precursor patterns at different temporal scales.
- **Boundary Detection Relevance:** **STRONGLY RELEVANT** — anomaly prediction is conceptually adjacent to boundary detection. Multi-resolution prediction captures regime changes at multiple scales, directly applicable to detecting boundaries of different granularities.

### C6. Koopman Invariants as Drivers of Emergent Time-Series Clustering in JEPAs
- **Authors:** Pablo Ruiz-Morales, Dries Vanoost, Davy Pissoort, Mathias Verbeke
- **Year:** 2025 | **Venue:** arXiv (2511.09783)
- **Key Contribution:** Theoretical analysis explaining why JEPAs cluster time-series by dynamical regimes. Proves idealized JEPA loss learns Koopman eigenfunctions (regime indicator functions). Key insight: near-identity linear predictor constraint forces interpretable regime-separating representations.
- **Boundary Detection Relevance:** **HIGHLY RELEVANT** — dynamical regime clustering IS activity segmentation. If JEPA naturally learns regime indicators, it provides a principled foundation for SSL-based boundary detection. Regime transitions = activity boundaries.

### C7. LeNEPA: No-Augmentation Next-Latent Prediction for Time-Series Representation Learning
- **Authors:** Alexander Chemeris, Ming Jin, Randall Balestriero
- **Year:** 2026 | **Venue:** arXiv (2607.00958)
- **Key Contribution:** Eliminates augmentation dependency in time-series SSL via next-latent-token prediction with causal backbone. Replaces stop-gradient/EMA with SIGReg isotropy regularization. Faster convergence (80% of final gain after 2-5k updates).
- **Boundary Detection Relevance:** Partially relevant — causal next-latent prediction naturally encodes temporal order, useful for detecting when predictions break down (i.e., boundaries).

### C8. CHARM (Multimodal JEPA): Giving Sensors a Voice — Multimodal JEPA for Semantic Time-Series Embeddings
- **Authors:** Utsav Dutta, Gerardo Pastrana, Sina Khoshfetrat Pakazad, Henrik Ohlsson
- **Year:** 2026 | **Venue:** arXiv (2605.31580)
- **Key Contribution:** Channel-aware transformer with textual channel descriptions, trained with JEPA + novel temporal stability loss. Latent-space prediction with description-aware gating for interpretability. Strong performance on classification, forecasting, anomaly detection via linear probe.
- **Boundary Detection Relevance:** Partially relevant — temporal stability loss could be adapted to detect when representations become unstable (at boundaries).

### C9. Action-Conditioned JEPAs: Beyond Patient Invariance — Learning Cardiac Dynamics
- **Authors:** Jose Geraldo Fernandes, Luiz Facury, Pedro Robles Dutenhefner, Wagner Meira
- **Year:** 2026 | **Venue:** arXiv (2604.22618)
- **Key Contribution:** Adapts LeJEPA to physiological time-series. Models pathology as transition vectors on latent state. Predicts future electrophysiological state given disease onset. Disentangles stable features from dynamic pathological forces. Outperforms supervised baselines on triage with superior sample efficiency.
- **Boundary Detection Relevance:** **HIGHLY RELEVANT** — models transitions/events as action-conditioned latent state changes, which is the core of boundary detection. Demonstrates that latent-space transition modeling captures clinically meaningful change points.

---

## D. SSL + Boundary Detection / Temporal Segmentation / Adaptive Windowing

### D1. TimePred: Efficient and Interpretable Offline Change Point Detection for High Volume Data
- **Authors:** Simon Leszek
- **Year:** 2025 | **Venue:** arXiv (2512.01562)
- **Key Contribution:** Self-supervised framework that reduces multivariate change-point detection to univariate mean-shift detection by predicting each sample's normalized time index. Supports XAI attribution. Reduces computational cost by 2 orders of magnitude.
- **Boundary Detection Relevance:** **DIRECTLY ADDRESSES** — this IS a change-point/boundary detection method built on SSL principles. The self-supervised time-index prediction pretext is directly transferable to HAR boundary detection.

### D2. CLaP: State Detection from Time Series
- **Authors:** Arik Ermshaus, Patrick Schäfer, Ulf Leser
- **Year:** 2025 | **Venue:** arXiv (2504.01783)
- **Key Contribution:** Novel self-supervised algorithm for Time Series State Detection (TSSD) — localizing and identifying latent states and their transitions. Uses cross-validated self-supervision to detect whether segments emerge from same state. Merges high-confusion segments.
- **Boundary Detection Relevance:** **DIRECTLY ADDRESSES** — TSSD is exactly activity segmentation/boundary detection. The self-supervised segment-similarity approach is directly applicable to HAR activity boundary localization.

### D3. Anomalous Change Point Detection Using Probabilistic Predictive Coding (PPC)
- **Authors:** Roelof G. Hup, Julian P. Merkofer, Alex A. Bhogal, Ruud J. G. van Sloun, Reinder Haakma, Rik Vullings
- **Year:** 2024 | **Venue:** arXiv (2405.15727)
- **Key Contribution:** Deep learning-based CPD/anomaly detection using probabilistic predictive coding. Jointly encodes sequential data to latent space and predicts subsequent representations with uncertainty. Linear time complexity, highly scalable.
- **Boundary Detection Relevance:** **DIRECTLY ADDRESSES** — predictive coding in latent space for change-point detection. The prediction error naturally spikes at boundaries where activity transitions occur. Directly applicable to HAR.

### D4. Enhancing Hierarchical Reinforcement Learning through Change Point Detection in Time Series
- **Authors:** Hemanath Arumugam, Falong Fan, Bo Liu
- **Year:** 2025 | **Venue:** arXiv (2510.24988)
- **Key Contribution:** Integrates self-supervised Transformer-based Change Point Detection into Option-Critic HRL framework. CPD trained with heuristic pseudo-labels to infer latent dynamics shifts. Uses change-points for option termination, behavioral cloning, and policy specialization.
- **Boundary Detection Relevance:** **DIRECTLY ADDRESSES** — self-supervised CPD for temporal segmentation. The approach of using intrinsic signals for pseudo-labels is transferable to HAR boundary detection.

### D5. Aircraft Trajectory Segmentation-based Contrastive Coding (ATSCC)
- **Authors:** Thaweerath Phisannupawong, Joshua Julian Damanik, Han-Lim Choi
- **Year:** 2024 | **Venue:** arXiv (2407.20028)
- **Key Contribution:** Self-supervised representation learning that leverages segmentability of trajectories. Ensures consistency within self-assigned segments via contrastive coding. Outperforms SOTA on classification and clustering.
- **Boundary Detection Relevance:** **DIRECTLY ADDRESSES** — combines segmentation with contrastive learning. The segmentation-based contrastive framework could be adapted for HAR where activity segments form natural contrastive units.

### D6. Time Series Representation Learning with Supervised Contrastive Temporal Transformer (SCOTT)
- **Authors:** Yuansan Liu, Sudanthi Wijewickrema, Christofer Bester, Stephen O'Leary, James Bailey
- **Year:** 2024 | **Venue:** arXiv (2403.10787)
- **Key Contribution:** Combines Transformer + TCN for global/local feature learning. Uses supervised contrastive loss for time series. **Explicitly evaluates on online Change Point Detection** — achieves ~98% AUPRC on HAR dataset and ~97% on surgical data.
- **Boundary Detection Relevance:** **DIRECTLY ADDRESSES** — demonstrates that contrastive representation learning enables excellent online change-point/boundary detection in HAR data.

### D7. Self-supervised New Activity Detection in Sensor-based Smart Environments
- **Authors:** (See arXiv 2401.10288)
- **Year:** 2024 | **Venue:** arXiv (2401.10288)
- **Key Contribution:** Self-supervised approach for detecting novel/unknown activities in smart environments. Contrastive-based detection of activity transitions to previously unseen classes.
- **Boundary Detection Relevance:** **DIRECTLY ADDRESSES** — novel activity detection inherently requires boundary awareness to identify when a new activity begins.

### D8. DISCOVER: Identifying Patterns of Daily Living in Human Activities from Smart Home Data
- **Authors:** (See arXiv 2503.01733)
- **Year:** 2025 | **Venue:** arXiv (2503.01733)
- **Key Contribution:** Self-supervised pattern discovery for daily living activities from smart home sensor data. Discovers activity structure without labels.
- **Boundary Detection Relevance:** Partially relevant — discovering activity patterns inherently involves segmenting continuous data into meaningful units.

### D9. Memory-Augmented LSTM Autoencoder for Unsupervised HAR with IMU Sensor Fusion
- **Authors:** (See arXiv 2606.28377)
- **Year:** 2026 | **Venue:** arXiv (2606.28377)
- **Key Contribution:** LSTM autoencoder with memory augmentation for unsupervised HAR via IMU sensor fusion. Addresses multi-sensor fusion and label dependency.
- **Boundary Detection Relevance:** Partially relevant — memory-augmented reconstruction could capture activity segment boundaries through reconstruction anomaly patterns.

---

## E. Additional Notable Papers (Surveys & Foundation Models)

### E1. Foundation Models Defining A New Era In Sensor-based HAR: A Survey And Outlook
- **Year:** 2026 | **Venue:** arXiv (2604.02711)
- **Key Contribution:** Comprehensive survey of foundation models for sensor-based HAR, covering SSL pretraining strategies, datasets, and future directions.

### E2. Scaling Laws in Wearable HAR
- **Year:** 2025 | **Venue:** arXiv (2502.03364)
- **Key Contribution:** Studies scaling laws for SSL in wearable HAR — how performance scales with data, model size, and compute.

### E3. COMODO: Cross-Modal Video-to-IMU Distillation for Efficient Egocentric HAR
- **Year:** 2025 | **Venue:** arXiv (2503.07259)
- **Key Contribution:** Cross-modal distillation from video to IMU for efficient egocentric HAR via self-supervised knowledge transfer.

### E4. Comparing Self-Supervised Learning Techniques for Wearable HAR
- **Year:** 2024 | **Venue:** arXiv (2404.15331)
- **Key Contribution:** Systematic comparison of SSL techniques (contrastive, masked, predictive) specifically for wearable HAR across multiple datasets.

### E5. Reducing Label Dependency in HAR with Wearables: From Supervised to Novel Weakly Self-Supervised Approaches
- **Year:** 2025 | **Venue:** arXiv (2512.19713)
- **Key Contribution:** Bridges supervised and weakly-supervised SSL approaches for HAR, charting the progression toward label-efficient learning.

---

## Summary Statistics

| Category | Papers Found |
|----------|-------------|
| Contrastive SSL for HAR | 14 |
| Predictive/Generative SSL for HAR | 14 |
| JEPA for TS/HAR | 9 |
| SSL + Boundary Detection/Segmentation | 9 |
| Surveys & Foundation Models | 5 |
| **Total Curated** | **51** |

### Papers Most Directly Relevant to Boundary Detection
1. **HAR-JEPA** (2607.16350) — JEPA for HAR with explicit evaluation on transitional activities
2. **Timestamp-supervised Segmentation** (2310.09114) — Joint segmentation + recognition with contrastive learning
3. **CLaP** (2504.01783) — Self-supervised state/segment detection
4. **TimePred** (2512.01562) — SSL-based change-point detection
5. **PPC** (2405.15727) — Probabilistic predictive coding for CPD
6. **SCOTT** (2403.10787) — Contrastive temporal transformer with CPD evaluation
7. **SC-JEPA** (2602.04643) — Multi-resolution JEPA for anomaly/boundary prediction
8. **Koopman-JEPA** (2511.09783) — Theoretical basis for regime clustering = segmentation
9. **HEPA** (2605.11130) — Horizon-conditioned event prediction
10. **CF-JEPA** (2606.07031) — Multi-horizon forward prediction preserving temporal continuity

### Key Gap Identified
**No paper (as of mid-2026) directly combines JEPA with explicit temporal boundary detection for wearable HAR.** The closest is HAR-JEPA (evaluates transitional activities) and the JEPA-for-CPD theoretical work (Koopman invariants). This represents a clear research opportunity.
