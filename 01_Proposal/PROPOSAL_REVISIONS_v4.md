# BASS-HAR Proposal — Supervisor Feedback Revisions (v4)

> **Date:** July 26, 2026
> **Supervisor:** Dr. Mohd Halim Mohd Noor
> **Student:** Mohammed Jasim
> **Document:** Summary of revisions applied per supervisor meeting feedback

---

## Revision Summary

| # | Supervisor Feedback | Action Taken |
|---|---|---|
| 1 | Remove SSL from the general objective | General objective rewritten — SSL is now the tool, not the goal |
| 2 | Remove "boundary-aware" and "context-aware" from objectives | Objectives rewritten to be technique-agnostic |
| 3 | Reduce to 3 objectives only (was 4) | Consolidated O3+O4 into a single evaluation objective |
| 4 | Stage 2 preprocessing: remove filtering, synchronization, sliding buffer | Only normalization retained |
| 5–9 | Read SSL papers, narrow to one SSL paradigm, update references to 2023+ | SSL section narrowed to **contrastive learning**; literature review written (separate document) |
| 10 | Literature review on SSL + Boundary Detection + GAP (due in 1 week) | Delivered as `SSL_Boundary_Detection_Literature_Review.md` |

---

## Revised §1.5 — Research Objectives

### General Objective

> Develop a framework for continuous human activity recognition using wearable sensors that addresses the problem of activity segmentation — specifically, discovering when activities begin and end in uninterrupted sensor streams — through self-supervised representation learning, and evaluating the framework's ability to recognize activities under varying degrees of label availability.

### Specific Objectives (3 only)

**O1 — Boundary Detection in Continuous Sensor Streams.**
Design a boundary detection mechanism that discovers activity transitions in continuous IMU data using SSL prediction errors, replacing the conventional fixed-size sliding-window approach with a data-driven method that eliminates the need for supervised boundary labels or handcrafted heuristics.

**O2 — Label-Efficient Activity Recognition.**
Develop a representation learning approach that reduces dependence on large labeled datasets by leveraging self-supervised learning on unlabeled sensor data, enabling effective activity recognition when only a small fraction of data is annotated.

**O3 — Comprehensive Evaluation.**
Evaluate the framework on multiple continuous-stream datasets using established protocols (Leave-One-Subject-Out cross-validation), comparing against relevant baselines including deep learning, Transformer-based, and self-supervised methods, with analysis of segmentation quality, recognition accuracy, and label efficiency.

---

## Revised §1.6 — Research Contributions

- **C1.** A segmentation mechanism for continuous HAR that discovers activity boundaries from sensor data, eliminating the fixed-window assumption.
- **C2.** A self-supervised representation learning approach that improves label efficiency for activity recognition, specifically by exploiting the temporal structure of activities.
- **C3.** A comprehensive evaluation framework with published, reproducible implementation, evaluated on multiple continuous-stream datasets with segmentation-aware and label-efficiency metrics.

---

## Revised §3.2 — Stage 1 — Data Acquisition and Preprocessing

Continuous IMU streams (tri-axial accelerometer and gyroscope) are resampled to a common sampling rate and **per-channel normalized** (z-score standardization per subject). No additional filtering, synchronization, or buffering is applied at this stage — the raw normalized signal is passed directly to the segmentation and representation learning modules. This preserves the full spectral content of the signal and avoids introducing artifacts that could affect boundary detection.

> **Change rationale (supervisor feedback #4):** Filtering (band-pass), timestamp synchronization, and sliding-buffer mechanisms were removed. Only normalization is retained.

---

## Revised SSL Scope (supervisor feedback #5–8)

### SSL Paradigm Selection: Contrastive Learning

The supervisor identified three SSL directions:
- **(A) Contrastive Learning** — constructs positive/negative pairs and learns invariant representations
- **(B) Predictive/Generative Pretext** — masked reconstruction, next-frame prediction
- **(C) JEPA** — joint-embedding predictive architecture (latent-space prediction without reconstruction)

**Selected paradigm: (A) Contrastive Learning — for immediate implementation.**

> **IMPORTANT UPDATE:** The supervisor (Mohd Halim Mohd Noor) and collaborator (Abdulrahman Baraka) just published **HAR-JEPA** [arXiv:2607.16350, July 17, 2026] — the first JEPA directly for sensor-based HAR. This paper:
> - Demonstrates JEPA's viability for HAR with an improved VICReg objective
> - Explicitly evaluates transitional activities (sit-to-stand, sit-to-lie)
> - Operates within the windowed classification paradigm (not boundary detection)
>
> **Strategic implication:** The proposed research can build directly on HAR-JEPA by extending it from windowed classification to **continuous-stream boundary detection**. Additionally, Koopman-JEPA theory [Ruiz-Morales et al., 2025] proves that JEPA naturally learns regime indicator functions — providing theoretical justification for why JEPA should excel at boundary detection.
>
> **Revised recommendation:** Consider JEPA (paradigm C) as the primary SSL direction, building on the supervisor's own HAR-JEPA work, supported by Koopman-JEPA theory. Contrastive learning remains the fallback with stronger existing empirical base.

Rationale for contrastive (near-term):
1. Most mature in the HAR domain — extensive recent literature (2023–2026) demonstrates strong results
2. Naturally suited to the boundary problem: temporal coherence (adjacent windows as positives) produces prediction errors that spike at activity transitions, providing an unsupervised boundary signal
3. Does not require pixel-level reconstruction (unlike masked modeling), which is ambiguous for IMU data
4. JEPA for time series is still nascent (first papers appearing 2026), making it higher-risk for a PhD

### Relationship Between SSL and Boundary Detection (supervisor feedback #8)

**Key principle: SSL operates FIRST on the continuous stream. Boundaries are discovered as a byproduct of the SSL objectives.**

The framework follows a **prediction-error-driven** design, not a sequential pipeline:

1. **SSL pre-training (Phase A, no boundaries needed):** The encoder and three SSL pretext heads (contrastive, continuity prediction, masked modeling) are trained directly on the continuous IMU stream. No boundary information is used. During this phase, prediction errors are recorded but not yet acted upon.

2. **Boundary discovery (Phase B, step 1):** The SSL prediction errors — spikes where the model fails to predict continuity or reconstruct masked regions — reveal activity transitions. These error signals are fed to the Boundary Refinement Module (BRM), which denoises them into sharp boundary probabilities.

3. **Feedback refinement (Phase B, steps 2–4):** The discovered boundaries are used to mine hard negatives for contrastive learning and align masks for masked modeling. This sharpens boundary sensitivity in the representations. Updated representations yield updated prediction errors, which yield updated boundaries. The loop iterates ~3 times until convergence.

4. **Classification (Phase C):** Boundary-localized context is used for final activity classification.

**Summary:**

| Role | Component | When |
|---|---|---|
| Representation learning on continuous stream | SSL pretext heads (TCL, CP, MTM) | Phase A |
| Boundary discovery via prediction errors | SSL prediction error → BRM | Phase B (step 1) |
| Boundary-informed hard negative mining | SSL pretext heads | Phase B (steps 2–3) |
| Final classification | Context-aware classifier | Phase C |

**This means:**
- SSL is **not** a post-hoc consumer of boundaries — it is the **primary boundary discovery mechanism**
- The BRM is **not** a detector — it is a denoiser/refiner of the SSL signal
- RuLSIF (statistical change-point detection) is retained only as an **optional fallback** (ablation), not the primary mechanism
- The research gap this addresses: no prior work uses SSL to discover boundaries in wearable sensors (OTAS demonstrated this only in video)

---

## Reference Update Policy (supervisor feedback #9)

All references in the revised proposal and literature review are from **2023 onwards**. Earlier foundational works (e.g., SimCLR 2020, BYOL 2020, Vaswani 2017) are cited only when defining fundamental concepts, and recent 2023+ papers that build upon them are preferred.

Key 2023+ references for the revised proposal:

1. Haresamudram et al. (2024). "Past, Present, and Future of Sensor-Based HAR Using Wearables: A Surveying Tutorial." arXiv:2411.14452.
2. Cai et al. (2026). "BenchHAR: Benchmarking SSL for Generalizable Sensor-based Activity Recognition." arXiv:2605.08296.
3. Wu et al. (2026). "Auto-Augmentation Contrastive Learning for Wearable-based HAR." arXiv:2602.02542.
4. Wang et al. (2023). "An Improved Masking Strategy for Self-Supervised Masked Reconstruction in HAR." arXiv:2312.04147.
5. Lee & Sim (2026). "CF-JEPA: Mask-free forward prediction for time-series representation learning." arXiv:2606.07031.
6. Bazarova et al. (2024). "Normalizing SSL for provably reliable Change Point Detection." arXiv:2410.13637.
7. Sheng & Huber (2025). "Reducing Label Dependency in HAR with Wearables." arXiv:2512.19713.
8. Nshimyimana et al. (2025). "PIM: Physics-Informed Multi-task Pre-training for HAR." arXiv:2503.17978.
9. Shamba et al. (2026). "Learning by Shifting: Temporal View Construction for Time Series Contrastive Learning." arXiv:2606.21957.
10. Bian et al. (2026). "Foundation Models Defining A New Era In Sensor-based HAR: A Survey." arXiv:2604.02711.

Full literature review: see `SSL_Boundary_Detection_Literature_Review.md`.
