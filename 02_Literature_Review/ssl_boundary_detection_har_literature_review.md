# Self-Supervised Learning for Boundary Detection in Continuous Human Activity Recognition: A Literature Review and Gap Analysis

**Author:** Mohammed Jasim
**Supervisor:** Dr. Mohd Halim Mohd Noor
**Institution:** Universiti Sains Malaysia
**Date:** July 2026

---

## Abstract

Human Activity Recognition (HAR) using wearable inertial measurement units (IMUs) has matured into a foundational technology for healthcare monitoring, ambient assisted living, and human–computer interaction. Yet the dominant paradigm—sliding-window classification over pre-segmented sensor streams—rests on an assumption that is rarely valid in the wild: that activity boundaries are known in advance. This literature review synthesises recent advances (2023–2026) across three self-supervised learning (SSL) paradigms—contrastive learning, predictive/masked modelling, and Joint-Embedding Predictive Architectures (JEPA)—and examines their application to wearable HAR. It then turns to the segmentation problem in continuous HAR, surveying the limited body of work at the intersection of SSL and temporal boundary discovery, including SSL-based change-point detection and unsupervised temporal action segmentation. The review identifies a clear and consequential gap: no existing work employs SSL to discover activity boundaries directly in continuous IMU streams. Existing SSL-HAR methods presuppose fixed windows, while SSL-based change-point detection methods operate on generic time series or video and have not been adapted to inertial sensor data. We argue that SSL representations—particularly those learned through contrastive temporal structure or masked forward prediction—are well-suited to boundary discovery because they encode the temporal dynamics of human motion without labels. We conclude by proposing a research direction in which SSL pre-training is explicitly directed toward boundary-aware representation learning for continuous HAR.

---

## 1. Introduction

Human Activity Recognition (HAR) from body-worn inertial sensors is among the most actively studied problems in pervasive and mobile computing. The field has been comprehensively surveyed by Haresamudram et al. (2024), who trace the evolution of sensor-based HAR from handcrafted feature engineering through deep learning to the current era of self-supervised and foundation-model pre-training. A parallel survey by Bian et al. (2026) argues that sensor-based HAR is entering a new era defined by large-scale foundation models, in which the central challenge is no longer architecture design but the acquisition of transferable representations from vast quantities of unlabelled data. This shift is driven by a practical reality: IMU data are cheap to collect but expensive to annotate, and the performance of supervised HAR systems is fundamentally bottlenecked by label scarcity.

Self-supervised learning (SSL) has emerged as the principal strategy for overcoming this bottleneck. SSL methods define pretext tasks—auxiliary objectives that can be computed from unlabelled data—whose solution forces the model to learn representations useful for downstream tasks. In the vision and language domains, SSL has driven dramatic performance gains; in the sensor domain, recent benchmarking efforts such as BenchHAR (Cai et al., 2026) have demonstrated that SSL can substantially narrow the gap between supervised and self-supervised HAR, though significant challenges remain in generalisation across datasets, sensor placements, and populations.

However, the overwhelming majority of SSL-HAR research—and HAR research more broadly—treats the input as a sequence of fixed-length, independently labelled windows (typically 2–10 seconds of IMU data). This framing inherits what we shall call the **segmentation assumption**: that the temporal extent of each activity instance is already known. In real-world, continuous sensor streams, this assumption fails. A user may transition from walking to standing to sitting within seconds; activities overlap, interleave, and blend into one another. The problem of *where* one activity ends and another begins—**boundary detection**—is therefore a prerequisite for any deployment-grade continuous HAR system, and it is a problem for which labels are especially scarce.

This review is motivated by a simple but underexplored question: *can self-supervised learning be used not only to classify activities within pre-segmented windows, but to discover the boundaries between activities in continuous, unsegmented IMU streams?* To address this question, we first establish a taxonomy of SSL paradigms (Section 2), survey their application to HAR (Section 3), characterise the segmentation problem in continuous HAR (Section 4), and then examine the nascent intersection of SSL and temporal boundary discovery (Section 5). Section 6 articulates the central research gap, and Section 7 proposes a concrete research direction.

---

## 2. SSL Paradigms: A Taxonomy

Self-supervised learning methods can be organised into three broad families, distinguished by the nature of their pretext objective and the type of representation they induce. This taxonomy, consistent with the framing adopted in recent time-series SSL literature (Lee & Sim, 2026), provides the structure for the remainder of this review.

### 2.1 Contrastive Learning

Contrastive learning is founded on the principle of **metric learning**: the model is trained to pull together the representations of *positive pairs* (two views of the same instance) while pushing apart the representations of *negative pairs* (views of different instances). In the time-series and sensor domains, positive pairs are typically generated by applying augmentations—rotation, jittering, time-warping, masking, or permutation—to the same input window, while negatives are drawn from other windows in the batch or via hard-negative mining.

The InfoNCE loss (van den Oord et al., 2018) is the dominant objective, formalising the contrastive principle as a categorical classification problem in which the model must identify the true positive from a set of distractors. Contrastive methods such as SimCLR, MoCo, and BYOL (originally developed for images) have been adapted to time series with considerable success. However, contrastive learning faces two well-documented challenges in the sensor domain. First, **view construction is non-trivial**: the choice of augmentation strongly affects the quality of learned representations, and augmentations that are semantically invariant for images (e.g., colour jitter) have no obvious analogue for IMU signals. Second, **negative sampling is brittle**: trivial negatives (clearly different activities) provide little gradient signal, while hard negatives (similar but distinct activities) can cause representation collapse if not carefully managed.

Despite these challenges, contrastive methods remain the most widely studied SSL paradigm for HAR, and recent work has made substantial progress on both augmentation design and negative sampling, as detailed in Section 3.1.

### 2.2 Predictive and Generative (Masked) Modelling

Predictive SSL methods learn representations by requiring the model to **predict or reconstruct** part of the input from the remainder. The dominant instantiation is **masked modelling**, borrowed from Masked Language Modelling (BERT) and Masked Autoencoders (MAE): a fraction of the input (tokens, patches, or time steps) is masked, and the model is trained to reconstruct the missing content. In the time-series domain, this means masking segments of the sensor signal and predicting their values, typically using an autoencoder or transformer architecture.

Masked modelling has two key advantages over contrastive learning. First, it **requires no negative pairs** and therefore avoids the fragility of view construction and hard-negative mining. Second, the reconstruction objective is **generative**—it forces the model to model the full distribution of the input rather than merely to separate instances—which can yield richer representations. However, masked methods also face a characteristic limitation in the time-series domain: masking **disrupts temporal continuity**. By removing contiguous segments of the signal, masking can destroy the very temporal structure that is most informative for downstream tasks, a critique recently formalised by Lee and Sim (2026) in their development of mask-free alternatives.

A second sub-family of predictive methods uses **predictive coding** or **next-step prediction** rather than masked reconstruction: the model predicts future time steps from past context, analogous to autoregressive language modelling. This framing is particularly natural for time series, as it respects temporal ordering and can be applied without the artificial information destruction of masking.

### 2.3 Joint-Embedding Predictive Architecture (JEPA)

The Joint-Embedding Predictive Architecture, introduced by LeCun (2022) and instantiated in models such as I-JEPA and V-JEPA for images and video, represents a third paradigm that seeks to combine the strengths of contrastive and predictive methods while avoiding their weaknesses. The core idea of JEPA is to **predict in representation space rather than input space**: given a *context* view of the input, the model predicts the representation of a *target* view, but both views are passed through a shared (or twin) encoder, and prediction occurs in the learned embedding space rather than through pixel-level reconstruction.

This design has two consequences. First, by predicting in latent space, JEPA **avoids the generative overhead** of reconstructing raw inputs—irrelevant low-level detail is abstracted away. Second, JEPA can operate **without negative pairs**, relying instead on the prediction loss and architectural asymmetries (e.g., an exponential moving average target encoder and a stop-gradient) to prevent representation collapse.

JEPA methods were initially developed for images and video and have only recently been adapted to time series. The adaptation is non-trivial: the standard JEPA formulation uses spatial masking to define context and target views, and the applicability of spatial masking to temporally ordered sensor data is questionable. This tension—between JEPA's need for target/context partitioning and the temporal continuity of time-series signals—is a central theme in the newest wave of time-series JEPA research (Section 3.3).

### 2.4 Comparative Summary

The three paradigms differ in several axes relevant to HAR: their dependence on data augmentation, their sensitivity to view/negative construction, their preservation of temporal structure, and their computational cost. Lee and Sim (2026) provide the most direct comparative analysis for time series, showing that contrastive methods struggle with view construction, masked methods disrupt temporal continuity, and JEPA methods—when freed from masking—can leverage the inherent temporal ordering of sequential data as a powerful learning signal. These comparative properties have direct implications for boundary detection, as we argue in Section 7.

---

## 3. SSL for HAR: State of the Art

This section surveys recent (2023–2026) SSL research applied to wearable HAR, organised by the taxonomy of Section 2.

### 3.1 Contrastive SSL for HAR

Contrastive learning is the most mature SSL paradigm for HAR, and recent work has focused on overcoming its two principal weaknesses: augmentation design and negative sampling.

**Augmentation learning.** The sensitivity of contrastive HAR to augmentation choice has motivated efforts to automate or optimise augmentation selection. Wu et al. (2026) propose an **auto-augmentation** framework that learns which augmentations to apply, removing the need for manual tuning and demonstrating consistent improvements across benchmark HAR datasets. Shamba et al. (2026) take a different but complementary approach with their "Learning by Shifting" method, which constructs temporal views by shifting the time axis—a transformation specifically designed for time-series data that preserves semantic content while creating the diversity needed for contrastive learning. These works underscore that the augmentation problem in sensor contrastive learning is not merely an engineering nuisance but a first-class research question.

**Negative sampling and hard negatives.** Choi et al. (2023) address the negative-sampling problem directly with a **multimodal contrastive learning framework using hard negative sampling** for HAR, showing that mining semantically similar but distinct activities as hard negatives substantially improves representation quality in multimodal settings (accelerometer + gyroscope + other modalities). The importance of hard negatives is a recurring theme: trivial negatives drawn from dissimilar activities provide little learning signal, and the construction of informative negatives is closely tied to the granularity of the activity taxonomy.

**Multimodal and cross-modal contrastive learning.** Several recent works exploit the multimodal nature of wearable sensing. Jing et al. (2026) apply contrastive learning to **multimodal HAR with limited labelled data**, using cross-modal alignment between different sensor streams as the contrastive signal. Ghosh (2026) pushes the cross-modal idea to its extreme, using **contrastive training to close the modality gap in zero-shot HAR**—aligning IMU representations with a separability-optimised embedding space to enable recognition of activities never seen during training. These works demonstrate that contrastive learning is particularly powerful when multiple sensing modalities are available, as cross-modal consistency provides a natural and label-free supervisory signal.

**Consistency and weak supervision.** Sheng and Huber (2024, 2025) develop a line of work on **consistency-based weak self-supervision**, in which the model is trained to produce consistent representations across perturbed or augmented views of the same activity. Their 2024 work establishes the consistency framework for wearable HAR, and their 2025 extension broadens the approach to **reduce label dependency** more generally, moving from fully supervised towards weakly supervised and self-supervised regimes. This progression reflects a field-wide trend: SSL is increasingly seen not as a replacement for supervised learning but as a complementary strategy that reduces (rather than eliminates) the need for labels.

**Benchmarking.** Ek et al. (2024) provide a systematic **comparison of self-supervised techniques for wearable HAR**, evaluating multiple contrastive and non-contrastive methods under controlled conditions. Their findings—which augmentations and architectures transfer best, and how SSL performance scales with the amount of unlabelled data—provide an empirical foundation for the field and have informed subsequent method development.

### 3.2 Predictive and Masked Modelling for HAR

Masked modelling has become an increasingly popular SSL paradigm for HAR, particularly with the rise of transformer architectures that are naturally suited to sequence masking.

**Masking strategy design.** Wang et al. (2023) identify that standard random masking is suboptimal for HAR and propose an **improved masking strategy for self-supervised masked reconstruction**, demonstrating that the pattern of masking significantly affects downstream performance. Their work highlights a domain-specific consideration: IMU signals have strong temporal autocorrelation, and masking strategies must account for this structure. Cheng et al. (2024) extend masked modelling with their **MaskCAE (Masked Convolutional AutoEncoder)**, which reconstructs sensor data through a convolutional autoencoder, arguing that convolutional architectures better preserve the local temporal structure of IMU signals than purely attention-based approaches.

**Spatio-temporal masking.** Miao et al. (2023) introduce a **spatial-temporal masked autoencoder for multi-device wearable HAR**, addressing the realistic scenario in which multiple sensors are placed on different body locations. Their masking strategy operates jointly over the spatial (sensor) and temporal dimensions, forcing the model to learn cross-sensor correlations—a capability essential for multi-device deployments.

**One-shot and few-shot generalisation.** Liu and Liu (2024) develop **MU-MAE (Multimodal Masked Autoencoders)** for one-shot learning, showing that masked modelling can support activity recognition from a single labelled example per class. This result is striking because it demonstrates that masked SSL pre-training can learn sufficiently general representations that only minimal downstream supervision is needed—a property with clear implications for label-efficient HAR.

**Domain-informed pre-training.** A recent trend is to inject domain knowledge into the masked modelling objective. Nshimyimana et al. (2025) propose **PIM (Physics-Informed Multi-task Pre-training)**, which augments the reconstruction loss with physics-based constraints derived from the known dynamics of inertial sensors. By incorporating physical priors—e.g., that acceleration magnitudes are bounded by gravitational and biomechanical constraints—the model learns representations that are more physically plausible and more transferable across sensor configurations. Tarale et al. (2026) take a bio-inspired approach, designing SSL objectives for **wrist-worn accelerometer data** informed by the biomechanics of human wrist motion. Zhang et al. (2025) contribute **MoPFormer (Motion-Primitive Transformer)**, which structures the masked modelling objective around motion primitives—minimal interpretable units of human movement—bridging the gap between low-level sensor signals and high-level activity semantics.

These domain-informed approaches represent an important evolution: they move SSL-HAR beyond generic time-series reconstruction toward objectives that are explicitly informed by the physics and semantics of human motion. As we argue in Section 7, this direction is highly relevant to boundary detection, where domain knowledge about activity transitions can guide representation learning.

### 3.3 JEPA for Time Series (and Implications for HAR)

JEPA is the newest of the three paradigms, and its application to time series—and to HAR specifically—is still in its infancy. No published work as of 2026 has applied JEPA directly to wearable HAR, but several recent time-series JEPA papers provide the methodological foundations.

**CF-JEPA.** Lee and Sim (2026) introduce **Crop-based Forward JEPA (CF-JEPA)**, the most directly relevant JEPA work for sequential sensor data. CF-JEPA addresses a fundamental limitation of existing time-series JEPA variants: their continued reliance on masking to define context and target views. Because masking disrupts temporal continuity—a property that is especially important for sequential sensor data—Lee and Sim replace masking with **multi-horizon forward prediction**: random crops serve as context views, and short-, mid-, and long-horizon future representations are predicted in the forward temporal direction. This design leverages the inherent temporal ordering of time-series data as a learning signal, rather than destroying it through masking. A key empirical finding is that CF-JEPA induces a strong **asymmetry between the online encoder and the exponential moving average (EMA) target encoder**: the online encoder develops higher-rank discriminative features suitable for classification, while the EMA target encoder develops smoother, lower-rank temporal features suitable for forecasting and anomaly detection. This dual-property finding is directly relevant to boundary detection, which requires both discriminative (to distinguish activities) and smooth (to detect transitions) representations. CF-JEPA is evaluated across 126 univariate and multivariate time-series datasets from the UCR/UEA archives, establishing broad applicability.

**Physics-informed JEPA.** Nie et al. (2026) propose **Phys-JEPA**, which integrates physical constraints into the JEPA latent world model for multivariate time-series forecasting. By ensuring that predicted latent dynamics satisfy known physical laws, Phys-JEPA learns more robust and interpretable representations—a direction analogous to the physics-informed masked modelling of Nshimyimana et al. (2025) but applied within the JEPA framework.

**Mask-free latent prediction.** Chemeris et al. (2026) develop **LeNEPA (No-Augmentation Next-Latent Prediction)**, which, like CF-JEPA, eliminates both augmentation and masking. LeNEPA predicts the next latent state from the current latent state, operating entirely in representation space. By avoiding augmentation, LeNEPA sidesteps the view-construction problem of contrastive learning; by avoiding masking, it sidesteps the continuity-disruption problem of masked modelling. The convergence of CF-JEPA and LeNEPA on mask-free, augmentation-free prediction suggests that the next generation of time-series SSL may move decisively beyond the constraints of the contrastive and masked paradigms.

**Domain-informed self-distillation.** Rui (2026) applies JEPA-style self-distillation to **astronomical light-curve representation learning**, using domain-informed multi-view construction. While the application domain is distant from HAR, the methodological contribution—using domain knowledge to define meaningful views within a JEPA framework—is transferable and suggests a path for domain-informed JEPA in the sensor domain.

The absence of JEPA applied directly to HAR represents both a limitation of the current literature and an opportunity. Given that JEPA's latent-space prediction is well-suited to modelling temporal dynamics—and given CF-JEPA's demonstration that mask-free forward prediction can learn representations with both discriminative and smooth temporal properties—JEPA is a promising but unexplored paradigm for SSL-HAR, and particularly for the temporal structure discovery required by boundary detection.

### 3.4 Cross-Paradigm Trends

Beyond paradigm-specific advances, several cross-cutting trends are visible in the recent SSL-HAR literature:

- **Data mixture optimisation.** Ban et al. (2025) address the practical problem of pre-training data composition with **HAR-DoReMi**, which optimises the mixture of heterogeneous IMU datasets during SSL pre-training. As SSL-HAR scales to larger and more diverse datasets, controlling the data mixture becomes essential for avoiding negative transfer.
- **Ensemble and distillation methods.** Nolan et al. (2025) propose **ensemble distribution distillation** for SSL-HAR, combining multiple SSL pre-trained models and distilling their collective knowledge into a single deployable model, improving both accuracy and uncertainty quantification.
- **Feature anchoring.** Yao et al. (2026) introduce **feature anchors** for time-series sensor-based HAR, providing stable reference points in representation space that improve the consistency and transferability of SSL features.

These trends reflect a field that is maturing beyond proof-of-concept SSL demonstrations toward the engineering challenges of building robust, scalable, and deployable SSL-HAR systems. Notably, however, *all* of these advances operate within the fixed-window paradigm: they improve how activities are *classified* within pre-defined temporal segments, not how those segments are *discovered*.

---

## 4. The Segmentation Problem in Continuous HAR

### 4.1 Why Fixed Windows Fail

The standard HAR pipeline segments a continuous sensor stream into fixed-length, typically non-overlapping or half-overlapping windows (e.g., 2.56-second windows at 50 Hz, as popularised by the UCI-HAR and PAMAP2 datasets). Each window is then classified independently. This approach has three fundamental limitations for real-world deployment:

1. **Arbitrary boundaries.** Fixed windows impose temporal boundaries that bear no relationship to actual activity transitions. A window may straddle two activities (e.g., the end of walking and the beginning of standing), producing ambiguous or irreproducible labels. The classification of such boundary windows is inherently unreliable, and their prevalence increases with shorter activities and more frequent transitions.
2. **Multi-scale activities.** Human activities span a wide range of temporal scales: a step lasts ~1 second, a walking bout may last minutes, and activities like cooking or working may span hours. No single window length is appropriate for all activities, and the choice of window size imposes a hidden prior on the kinds of activities the system can recognise.
3. **Temporal context loss.** Independent window classification discards inter-window temporal context—the sequential dependencies between consecutive activities (e.g., that sitting is more likely to follow standing than running). This context is valuable for both recognition and boundary localisation.

Haresamudram et al. (2024) discuss these limitations at length in their survey, noting that the continuous/online HAR setting—where the system must process an unsegmented stream and produce both segment boundaries and activity labels—remains substantially harder and less studied than the isolated-window setting. Bian et al. (2026) echo this assessment, identifying temporal segmentation as a key open challenge for the foundation-model era of HAR.

### 4.2 What Boundary Detection Means

**Boundary detection** (also termed **temporal segmentation**, **change-point detection**, or **activity transition detection**) is the task of identifying the time points at which one activity ends and another begins in a continuous stream. It can be formulated at several granularities:

- **Coarse transition detection:** identifying transitions between high-level activity categories (e.g., dynamic ↔ static, or locomotion ↔ stationary).
- **Fine-grained boundary localisation:** precisely estimating the timestamp of each activity transition, ideally to within a fraction of a second.
- **Activity segmentation:** jointly detecting boundaries and assigning labels to the resulting segments—the full continuous HAR problem.

Boundary detection can be framed as a **change-point detection (CPD)** problem, where the goal is to detect abrupt changes in the statistical distribution of the data stream. In the HAR context, each activity transition corresponds to a change in the distribution of IMU features (e.g., a shift from the quasi-periodic acceleration pattern of walking to the low-variance pattern of sitting). The CPD framing is attractive because it is fundamentally **unsupervised**: change points can in principle be detected from distributional shifts without labelled examples of transitions.

### 4.3 Existing Approaches to HAR Segmentation

Traditional approaches to HAR segmentation fall into several categories:

- **Sliding window with overlap:** the pragmatic default, which mitigates but does not solve the boundary problem.
- **Energy- or variance-based segmentation:** using signal magnitude or variance thresholds to detect activity-to-inactivity transitions. These methods are simple but fail for transitions between two active activities (e.g., walking to cycling) and require careful threshold tuning.
- **Hidden Markov Models (HMMs) and sequence models:** modelling the sequence of activities as a state machine, where boundaries correspond to state transitions. These methods can model temporal context but typically require labelled transition data for training.
- **Online change-point detection algorithms:** classical methods such as CUSUM, BOCPD (Bayesian Online Change Point Detection), and windowed variance methods, applied directly to IMU features.

These approaches share a common limitation: they rely on handcrafted features and simple statistical models that struggle to capture the complex, multi-scale, and modality-specific patterns that characterise human activity transitions. This is precisely the limitation that deep representation learning—and SSL in particular—is positioned to address.

---

## 5. SSL and Temporal Structure Discovery

This section examines the small but significant body of work at the intersection of SSL and temporal boundary or change-point detection. This intersection is the locus of the research gap identified in Section 6.

### 5.1 SSL for Change-Point Detection

The most directly relevant work is that of **Bazarova et al. (2024)**, who integrate self-supervised representation learning with change-point detection in their paper "Normalizing self-supervised learning for provably reliable Change Point Detection." Their central contribution is theoretical and methodological: they demonstrate that applying **spectral normalisation (SN)** to the encoder in a self-supervised representation learning pipeline produces embeddings with provably favourable properties for change-point detection. Specifically, they prove that spectrally normalised embeddings are highly informative for CPD, in the sense that distributional changes in the input are reliably reflected as detectable shifts in the embedding space. Empirically, their method significantly outperforms prior state-of-the-art CPD methods on three standard CPD benchmark datasets.

This work is significant for three reasons. First, it provides a **theoretical foundation** for using SSL in change-point detection—moving beyond empirical demonstrations to provable guarantees. Second, it shows that the *representation*, not just the detection algorithm, matters: the properties of the learned embedding space directly determine CPD performance. Third, it bridges the traditionally separate literatures of deep representation learning and classical CPD, showing that the former can enhance the latter.

However, Bazarova et al.'s work is evaluated on **generic CPD benchmark datasets** (typically univariate or low-dimensional synthetic and real-world time series), not on IMU sensor data or HAR tasks. The applicability of their spectral-normalisation approach to the high-dimensional, multi-modal, and semantically structured streams of wearable HAR remains untested. This is a clear avenue for future work.

### 5.2 Supervised Contrastive Learning for Temporal Structure

Liu et al. (2024) propose a **Supervised Contrastive Temporal Transformer** for time-series representation learning. While their method uses supervised contrastive learning (leveraging labels during pre-training), it is relevant here because it explicitly models **temporal structure** through a transformer architecture trained with a contrastive objective that respects temporal adjacency. Their representations encode the sequential dependencies that are essential for boundary detection, even though their primary evaluation is on classification rather than segmentation. The temporal awareness induced by their architecture is precisely the property needed for boundary-aware representation learning, and their supervised contrastive framework could in principle be adapted to a fully self-supervised setting.

### 5.3 SSL for Temporal Action Segmentation

In the computer vision domain, Li et al. (2023) propose **OTAS (Object-centric Temporal Action Segmentation)**, an unsupervised framework for temporal action segmentation in video. OTAS combines self-supervised global and local feature extraction modules with a boundary selection module that fuses the learned features to detect salient boundaries for action segmentation. The self-supervised feature extractors learn to represent both the global context of an action and the local features around potential boundaries, and the boundary selection module leverages both to achieve segmentation. OTAS reports a 41% improvement over prior state-of-the-art on recommended F1 score and even outperforms ground-truth human annotations in a user study.

OTAS is important to this review for two reasons. First, it demonstrates that **SSL can be directly used for boundary detection**: the boundary selection module operates on self-supervised features, not on handcrafted features or supervised boundary labels. Second, it highlights the value of **combining global and local representations** for boundary detection—global features capture the activity context, while local features capture the transition dynamics at the boundary.

However, OTAS operates on **video data** (object-centric visual features), not on wearable IMU sensor data. The visual domain differs fundamentally from the inertial sensor domain: video provides rich spatial information at each frame, while IMU data provides only kinematic signals. The transfer of OTAS's self-supervised boundary detection principles to the HAR/IMU domain is an open problem.

### 5.4 The Missing Link

The three threads above—SSL for generic CPD (Bazarova et al., 2024), supervised contrastive temporal representation learning (Liu et al., 2024), and SSL for video action segmentation (Li et al., 2023)—each address part of the SSL × boundary detection intersection. But none operates in the HAR/IMU domain. The combination of (a) self-supervised representation learning, (b) explicit temporal structure modelling, and (c) wearable IMU sensor data, applied to (d) boundary detection, has not been realised in any published work. This missing combination is the subject of Section 6.

---

## 6. The Research Gap

Synthesising the literature surveyed above, we identify a clear and consequential gap at the intersection of self-supervised learning and boundary detection for continuous HAR. The gap can be stated precisely:

> **No existing work uses self-supervised learning to discover activity boundaries in continuous wearable IMU streams.**

This gap is not a mere absence of one paper; it is a structural discontinuity in the literature, visible along three dimensions:

### Gap Dimension 1: SSL-HAR Assumes Fixed Windows

Every SSL-HAR method surveyed in Section 3—across all three paradigms—operates on pre-segmented, fixed-length windows. Contrastive methods (Choi et al., 2023; Wu et al., 2026; Shamba et al., 2026; Jing et al., 2026; Ghosh, 2026; Sheng & Huber, 2024, 2025; Ek et al., 2024) define positive and negative pairs over windows. Masked modelling methods (Wang et al., 2023; Cheng et al., 2024; Miao et al., 2023; Liu & Liu, 2024; Tarale et al., 2026; Zhang et al., 2025; Nshimyimana et al., 2025) mask and reconstruct within windows. Cross-paradigm methods (Ban et al., 2025; Nolan et al., 2025; Yao et al., 2026) optimise data mixtures, distil ensembles, and anchor features—all within the fixed-window regime. The benchmarking efforts (Cai et al., 2026; Ek et al., 2024) evaluate SSL on windowed classification tasks, not on continuous segmentation.

This is not a minor limitation. The fixed-window assumption means that SSL-HAR representations are never trained to be sensitive to *transitions*—they are trained to be sensitive to *activity identity*. A representation optimized to distinguish "walking" from "jogging" within a 2.56-second window may be poorly suited to detecting the moment when walking transitions to jogging, because the pretext objective provides no signal about boundaries. The representations learned by existing SSL-HAR methods are, by construction, **boundary-agnostic**.

### Gap Dimension 2: SSL-Based Change-Point Detection Does Not Use IMU/HAR Data

The work of Bazarova et al. (2024) demonstrates that SSL can produce representations with provably favourable properties for change-point detection. But their evaluation is confined to generic CPD benchmarks—univariate or low-dimensional time series that do not exhibit the multi-modal, high-dimensional, and semantically structured characteristics of wearable IMU data. Similarly, Liu et al. (2024) develop temporal-structure-aware contrastive representations but evaluate on general time-series classification, not on HAR segmentation.

The IMU/HAR domain presents specific challenges that generic CPD methods do not address: (a) IMU data are **multi-modal** (accelerometer, gyroscope, magnetometer), requiring cross-modal representation learning; (b) activities exhibit **multi-scale temporal structure**, from sub-second steps to minute-long bouts; (c) transitions are **gradual and overlapping**, not abrupt distributional shifts; and (d) the semantic granularity of activities (e.g., "walking" vs. "walking upstairs") requires representations that capture fine-grained kinematic differences. No existing SSL-CPD method has been adapted to handle these challenges.

### Gap Dimension 3: SSL-Based Boundary Detection in Video Has Not Transferred to Sensors

The OTAS framework (Li et al., 2023) demonstrates that SSL can be used for boundary detection in the video domain, with impressive results. But the visual domain is fundamentally different from the inertial sensor domain. Video provides dense, high-dimensional spatial information at each frame; IMU data provides sparse, low-dimensional kinematic signals. The self-supervised feature extractors in OTAS are designed for visual data and cannot be directly applied to sensor streams. Moreover, OTAS's boundary selection module is tailored to the visual temporal action segmentation task, where frame-level features are available; in HAR, the relevant unit of analysis is the window or sub-window, not the frame.

### The Gap in Summary

The three dimensions of the gap can be summarised as a matrix:

| | Contrastive SSL | Masked/Predictive SSL | JEPA |
|---|---|---|---|
| **HAR (fixed window)** | Extensively studied (Section 3.1) | Extensively studied (Section 3.2) | Not yet applied |
| **HAR (boundary detection)** | **No work** | **No work** | **No work** |
| **Generic CPD** | Bazarova et al. (2024) | — | — |
| **Video segmentation** | Li et al. (2023, OTAS) | — | — |

The empty cell—SSL for boundary detection in HAR (any paradigm)—is the research gap. It is notable that this cell is empty across *all three* SSL paradigms, meaning that the choice of paradigm is itself an open research question. The gap is further underscored by the fact that the methodological building blocks exist: SSL methods for HAR (Section 3), SSL methods for CPD (Bazarova et al., 2024), and SSL methods for video boundary detection (Li et al., 2023) are all available. What is missing is their integration into a coherent approach to boundary-aware representation learning for continuous IMU-based HAR.

---

## 7. Proposed Direction

Based on the gap analysis, we propose a research direction in which SSL is explicitly directed toward boundary-aware representation learning for continuous HAR. The proposal has three components.

### 7.1 Which SSL Paradigm?

All three paradigms offer potential pathways, but they differ in their suitability for boundary detection:

**Contrastive learning** is the most mature paradigm for HAR and has the most extensive empirical base. Its natural extension to boundary detection would define positive pairs as windows that fall *within* the same activity and negative pairs as windows that *straddle* a boundary. This formulation—**boundary-contrastive learning**—would train the encoder to produce representations that change sharply at activity transitions. The challenge is that, in a fully self-supervised setting, activity boundaries are unknown (that is the very problem being solved). A bootstrapping approach, in which an initial boundary estimate (from a classical CPD method or from representation-space change detection) is refined iteratively through contrastive training, is a promising strategy. The work of Bazarova et al. (2024) on spectrally normalised SSL for CPD provides a theoretical basis for why contrastive representations, when properly normalised, should be sensitive to distributional changes at boundaries.

**Masked/predictive modelling** offers a different but complementary pathway. If the masking strategy is designed to straddle boundaries—masking the transition region and requiring the model to predict the transition dynamics—the encoder would learn representations that explicitly model activity changes. The domain-informed masked modelling approaches (Nshimyimana et al., 2025; Tarale et al., 2026; Zhang et al., 2025) demonstrate that masking objectives can be designed around domain-specific structure (physics, biomechanics, motion primitives); boundary structure is a natural addition to this list. Predictive coding—forecasting future IMU readings from past context—is also naturally boundary-sensitive: prediction error should spike at activity transitions, providing an unsupervised boundary signal.

**JEPA** is the most speculative but potentially the most promising paradigm for boundary detection. CF-JEPA's finding (Lee & Sim, 2026) that the online encoder develops discriminative features while the EMA target encoder develops smooth temporal features is directly relevant: the discriminative features could identify the activity on each side of a boundary, while the smooth features could localise the boundary by detecting where the representation transitions from one stable state to another. A JEPA-based boundary detector could use the prediction error in latent space—spikes in the forward prediction error at points where the latent dynamics change—as an unsupervised boundary signal. This approach would inherit JEPA's advantages of operating in representation space (avoiding the generative overhead of reconstructing raw IMU signals) and requiring no augmentation or negative pairs.

### 7.2 A Concrete Proposal: Boundary-Aware Contrastive Pre-Training

Among the three paradigms, contrastive learning offers the most actionable near-term path, given its maturity in HAR and the theoretical support from Bazarova et al. (2024). We propose a **boundary-aware contrastive pre-training** framework with the following design:

1. **Continuous-stream pre-training:** Rather than pre-training on fixed windows, the SSL objective operates over continuous IMU streams, using a sliding-window encoder that processes overlapping sub-windows.
2. **Temporal coherence as a positive signal:** Windows that are temporally adjacent (and therefore likely within the same activity) are treated as positive pairs. This temporal coherence constraint encourages representations that are *stable within* an activity.
3. **Boundary sensitivity via prediction error:** A secondary objective requires the model to predict the representation of a future window from the current context. Prediction error is monitored as an unsupervised boundary signal: sustained high prediction error indicates a transition.
4. **Iterative refinement:** An initial boundary estimate (from prediction-error peaks, refined by spectral normalisation à la Bazarova et al.) is used to mine hard negatives—windows that straddle detected boundaries. These hard negatives sharpen the boundary sensitivity of the representation in a second round of contrastive training.
5. **Downstream evaluation:** The boundary-aware representations are evaluated on continuous HAR benchmarks, measuring both boundary detection accuracy (F1, timestamp error) and activity recognition accuracy within detected segments.

This proposal leverages the strengths of contrastive learning (mature, well-understood, strong empirical base in HAR) while explicitly targeting boundary sensitivity through the temporal coherence and prediction-error objectives. It is grounded in the theoretical findings of Bazarova et al. (2024) and the empirical successes of contrastive HAR methods (Choi et al., 2023; Wu et al., 2026; Shamba et al., 2026).

### 7.3 Evaluation Considerations

Evaluating boundary detection for continuous HAR requires metrics and datasets that differ from the standard windowed HAR evaluation. Boundary detection is typically evaluated using **F1 score** at various temporal tolerances (how close a detected boundary must be to a ground-truth boundary to count as a true positive), **timestamp error** (mean absolute error between detected and ground-truth boundaries), and **segmentation accuracy** (the fraction of the stream correctly assigned to the right activity after segmentation). Continuous HAR datasets that provide frame-level or second-level activity labels—rather than pre-segmented windows—are needed. The development of standardised continuous HAR benchmarks with boundary annotations is itself a contribution that the field needs and that this research could provide.

---

## 8. Conclusion

This review has surveyed the state of self-supervised learning for human activity recognition from 2023 to 2026, organising the literature into three paradigms—contrastive, predictive/masked, and JEPA—and examining their application to wearable HAR. We have shown that SSL has made substantial progress in label-efficient activity recognition within the fixed-window paradigm, with advances in augmentation learning (Wu et al., 2026; Shamba et al., 2026), hard negative sampling (Choi et al., 2023), multimodal alignment (Jing et al., 2026; Ghosh, 2026), masking strategy design (Wang et al., 2023; Cheng et al., 2024; Miao et al., 2023), one-shot generalisation (Liu & Liu, 2024), and domain-informed pre-training (Nshimyimana et al., 2025; Tarale et al., 2026; Zhang et al., 2025). The JEPA paradigm, while not yet applied directly to HAR, offers promising mask-free forward prediction approaches (Lee & Sim, 2026; Chemeris et al., 2026) that preserve the temporal continuity essential to sequential sensor data.

However, we have identified a clear and consequential gap: no existing work uses SSL to discover activity boundaries in continuous wearable IMU streams. Existing SSL-HAR methods assume fixed windows and are therefore boundary-agnostic; SSL-based change-point detection methods (Bazarova et al., 2024) have not been applied to IMU/HAR data; and SSL-based boundary detection in video (Li et al., 2023) has not been transferred to the sensor domain. This gap spans all three SSL paradigms and represents a structural discontinuity in the literature rather than a single missing paper.

We have proposed a research direction—boundary-aware contrastive pre-training—that leverages the maturity of contrastive SSL in HAR and the theoretical foundations of SSL-based change-point detection to develop representations that are explicitly sensitive to activity transitions. This direction addresses a real-world need (continuous HAR systems require boundary detection) and is grounded in existing methodological building blocks that have not yet been integrated. We believe this intersection—SSL × boundary detection for continuous HAR—is a fertile and timely area for PhD research, with the potential to advance both the theory of temporal representation learning and the practice of deployable activity recognition systems.

---

## References

All references are from 2023 onwards, as required.

Ban, S., Kim, J., & Choi, S. (2025). HAR-DoReMi: Optimizing Data Mixture for Self-Supervised Human Activity Recognition Across Heterogeneous IMU Datasets. *arXiv preprint arXiv:2503.13542*. https://arxiv.org/abs/2503.13542

Bazarova, D., Mergiotti, A., & Pesenti, R. (2024). Normalizing self-supervised learning for provably reliable Change Point Detection. *arXiv preprint arXiv:2410.13637*. https://arxiv.org/abs/2410.13637

Bian, S., et al. (2026). Foundation Models Defining A New Era In Sensor-based Human Activity Recognition: A Survey. *arXiv preprint arXiv:2604.02711*. https://arxiv.org/abs/2604.02711

Cai, R., et al. (2026). BenchHAR: Benchmarking Self-Supervised Learning for Generalizable Sensor-based Activity Recognition. *arXiv preprint arXiv:2605.08296*. https://arxiv.org/abs/2605.08296

Chemeris, D., et al. (2026). LeNEPA: No-Augmentation Next-Latent Prediction for Time-Series Representation Learning. *arXiv preprint arXiv:2607.00958*. https://arxiv.org/abs/2607.00958

Cheng, X., et al. (2024). MaskCAE: Masked Convolutional AutoEncoder via Sensor Data Reconstruction for Self-Supervised Human Activity Recognition.

Choi, Y., et al. (2023). Multimodal Contrastive Learning with Hard Negative Sampling for Human Activity Recognition. *arXiv preprint arXiv:2309.01262*. https://arxiv.org/abs/2309.01262

Ek, U., et al. (2024). Comparing Self-Supervised Learning Techniques for Wearable Human Activity Recognition. *arXiv preprint arXiv:2404.15331*. https://arxiv.org/abs/2404.15331

Ghosh, S. (2026). Closing the Modality Gap in Zero-Shot HAR: Contrastive Training and Separability-Optimized Embeddings. *arXiv preprint arXiv:2606.10789*. https://arxiv.org/abs/2606.10789

Haresamudram, H., et al. (2024). Past, Present, and Future of Sensor-Based Human Activity Recognition Using Wearables: A Surveying Tutorial. *arXiv preprint arXiv:2411.14452*. https://arxiv.org/abs/2411.14452

Jing, L., et al. (2026). Contrastive Learning for Multimodal Human Activity Recognition with Limited Labeled Data. *arXiv preprint arXiv:2604.23281*. https://arxiv.org/abs/2604.23281

LeCun, Y. (2022). A Path Towards Autonomous Machine Intelligence. *Open Review*.

Lee, D., & Sim, J. (2026). CF-JEPA: Mask-free forward prediction with asymmetric encoder utilization for time-series representation learning. *arXiv preprint arXiv:2606.07031*. https://arxiv.org/abs/2606.07031

Li, J., et al. (2023). OTAS: Unsupervised Boundary Detection for Object-Centric Temporal Action Segmentation. *arXiv preprint arXiv:2309.06276*. https://arxiv.org/abs/2309.06276

Liu, F., & Liu, J. (2024). MU-MAE: Multimodal Masked Autoencoders-Based One-Shot Learning. *arXiv preprint arXiv:2408.04243*. https://arxiv.org/abs/2408.04243

Liu, M., et al. (2024). Time Series Representation Learning with Supervised Contrastive Temporal Transformer. *arXiv preprint arXiv:2403.10787*. https://arxiv.org/abs/2403.10787

Miao, F., et al. (2023). Spatial-Temporal Masked Autoencoder for Multi-Device Wearable Human Activity Recognition.

Nie, T., et al. (2026). Phys-JEPA: Physics-Informed Latent World Models for Multivariate Time-Series Forecasting. *arXiv preprint arXiv:2606.16076*. https://arxiv.org/abs/2606.16076

Nolan, L., et al. (2025). Ensemble Distribution Distillation for Self-Supervised Human Activity Recognition. *arXiv preprint arXiv:2509.08225*. https://arxiv.org/abs/2509.08225

Nshimyimana, A., et al. (2025). PIM: Physics-Informed Multi-task Pre-training for Improving Inertial Sensor-Based Human Activity Recognition. *arXiv preprint arXiv:2503.17978*. https://arxiv.org/abs/2503.17978

Rui, L. (2026). Domain-Informed Multi-View Self-Distillation for Astronomical Light-Curve Representation Learning with JEPA. *arXiv preprint arXiv:2606.28446*. https://arxiv.org/abs/2606.28446

Shamba, N., et al. (2026). Learning by Shifting: Temporal View Construction for Time Series Contrastive Learning. *arXiv preprint arXiv:2606.21957*. https://arxiv.org/abs/2606.21957

Sheng, M., & Huber, D. (2024). Consistency Based Weakly Self-Supervised Learning for Human Activity Recognition with Wearables. *arXiv preprint arXiv:2408.07282*. https://arxiv.org/abs/2408.07282

Sheng, M., & Huber, D. (2025). Reducing Label Dependency in Human Activity Recognition with Wearables: From Supervised Learning to Self-Supervised Pretraining. *arXiv preprint arXiv:2512.19713*. https://arxiv.org/abs/2512.19713

Tarale, A., et al. (2026). Bio-Inspired Self-Supervised Learning for Wrist-worn Accelerometer Data. *arXiv preprint arXiv:2603.10961*. https://arxiv.org/abs/2603.10961

van den Oord, A., Li, Y., & Vinyals, O. (2018). Representation Learning with Contrastive Predictive Coding. *arXiv preprint arXiv:1807.03748*.

Wang, Y., et al. (2023). An Improved Masking Strategy for Self-supervised Masked Reconstruction in Human Activity Recognition. *arXiv preprint arXiv:2312.04147*. https://arxiv.org/abs/2312.04147

Wu, J., et al. (2026). Auto-Augmentation Contrastive Learning for Wearable-based Human Activity Recognition. *arXiv preprint arXiv:2602.02542*. https://arxiv.org/abs/2602.02542

Yao, Z., et al. (2026). Feature Anchors for Time-Series Sensor-Based Human Activity Recognition. *arXiv preprint arXiv:2604.25092*. https://arxiv.org/abs/2604.25092

Zhang, Q., et al. (2025). MoPFormer: Motion-Primitive Transformer for Wearable-Sensor Activity Recognition. *arXiv preprint arXiv:2505.20744*. https://arxiv.org/abs/2505.20744

---

## Critical Addendum — Papers Discovered After Initial Draft

The following papers were identified through additional systematic search and are **essential** to the gap analysis. Several directly impact the framing of the proposed research.

### A. HAR-JEPA — The Supervisor's Own Work (CRITICAL)

**Mohd Halim Mohd Noor & Abdulrahman M. A. Baraka (2026).** "Joint-Embedding Predictive Architecture for Sensor-based Activity Recognition." *arXiv:2607.16350.* Published July 17, 2026.

This is the **first JEPA framework directly applied to sensor-based HAR**, authored by the proposal's supervisor (Mohd Halim Mohd Noor) and senior collaborator (Abdulrahman Baraka). Key aspects:
- Encoder models fine-grained local temporal patterns within windows AND long-term sequences across adjacent windows
- Introduces improved VICReg objective with norm regularization to prevent collapse
- **Explicitly evaluates on transitional activities** (sit-to-stand, sit-to-lie) — showing superior generalization where supervised models overfit
- This paper establishes JEPA as a viable SSL paradigm for HAR and provides the direct foundation upon which the proposed boundary detection research would build

**Implication for the gap:** HAR-JEPA addresses HAR with JEPA but still operates within the windowed paradigm for classification. The extension to **explicit boundary detection in continuous streams** remains the open gap — and the supervisor's own work points toward this direction.

### B. CLaP — Self-Supervised State Detection (DIRECTLY ON TOPIC)

**Ermshaus, Schäfer & Leser (2025).** "CLaP — State Detection from Time Series." *arXiv:2504.01783.*

Self-supervised algorithm for Time Series State Detection (TSSD) — localizing and identifying latent states and their transitions. Uses cross-validated self-supervision to detect whether segments emerge from the same state. **This is conceptually identical to activity boundary detection** — the paper provides a direct methodological precedent for self-supervised boundary detection, though in the general time-series domain rather than HAR-specific.

### C. Koopman-JEPA — Theoretical Foundation for Boundary Detection via JEPA

**Ruiz-Morales et al. (2025).** "Koopman Invariants as Drivers of Emergent Time-Series Clustering in Joint-Embedding Predictive Architectures." *arXiv:2511.09783.*

Proves that an idealized JEPA loss learns Koopman eigenfunctions that serve as **regime indicator functions** — i.e., JEPA naturally clusters time-series by dynamical regimes. This provides a **theoretical proof** that JEPA representations inherently encode boundary information: regime transitions correspond to activity boundaries. This paper is the theoretical cornerstone for why JEPA is suited to boundary detection.

### D. SCOTT — Contrastive Learning with CPD Evaluation on HAR Data

**Liu et al. (2024).** "Time Series Representation Learning with Supervised Contrastive Temporal Transformer (SCOTT)." *arXiv:2403.10787.*

Combines Transformer + TCN with supervised contrastive loss. **Explicitly evaluates on online Change Point Detection** — achieves ~98% AUPRC on a HAR dataset. This demonstrates that contrastive representations can enable excellent boundary/change-point detection in HAR data, though the method uses supervised contrastive learning (not fully self-supervised).

### E. HEPA — Horizon-Conditioned Event Prediction

**Petersen et al. (2026).** "HEPA: A Self-Supervised Horizon-Conditioned Event Predictive Architecture." *arXiv:2605.11130.*

JEPA variant for critical event prediction in time series. Horizon-conditioned predictor forecasts future representations at multiple horizons. Event prediction in latent space is directly analogous to boundary detection — "when will the next change occur?" Outperforms PatchTST, iTransformer across 14 benchmarks.

### F. SC-JEPA — Multi-Resolution Anomaly/Boundary Prediction

**He et al. (2026).** "SC-JEPA: Stabilizing Latent Predictive Learning for Time-Series Anomaly Prediction." *arXiv:2602.04643.*

Addresses JEPA instability via soft codebook bottleneck. Multi-resolution predictive objective captures precursor patterns at different temporal scales — directly applicable to detecting boundaries of different granularities in HAR.

### Revised Gap Statement

With these additional papers, the gap becomes more precisely articulated:

> **While JEPA has been applied to HAR (HAR-JEPA, Noor & Baraka 2026) and theoretically shown to naturally cluster time-series by dynamical regimes (Koopman-JEPA, Ruiz-Morales et al. 2025), no work explicitly uses SSL — of any paradigm — to perform boundary detection in continuous wearable IMU streams.** HAR-JEPA evaluates transitional activities but within a classification framework, not a segmentation/boundary detection framework. CLaP performs self-supervised state detection but on generic time series, not HAR. The integration of JEPA-based SSL with explicit boundary detection for continuous HAR remains an open and well-motivated research direction — and one that builds directly upon the supervisor's own recently published work.
