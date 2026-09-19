# BASS-HAR Research Ideas: PhD-Level Extension Directions
## Comprehensive Literature Review & Gap Analysis (arXiv 2023–2026)

**Context:** BASS-HAR = Boundary-Aware Self-Supervised HAR using IMU/wearable sensors. SSL (contrastive, continuity prediction, masked modeling) operates on continuous streams; prediction errors reveal activity boundaries; BRM refines them. Core gap: no prior work uses SSL to discover boundaries in wearable sensors (OTAS did it in video only).

**Student:** Mohammed Jasim, USM, supervisor Dr. Mohd Halim Mohd Noor (HAR-JEPA, arXiv:2607.16350)

---

## AREA 1: Self-Supervised Boundary Detection in Time-Series/Sensor Data

### What's Been Done
- **TS-CP²** (Deldari et al., 2020, arXiv:2011.14097): Self-supervised change point detection using Contrastive Predictive Coding. Learns embeddings that separate time-adjacent intervals from cross-time intervals. **Closest existing work to BASS-HAR's core concept**, but: (a) uses generic time series, not wearable/IMU; (b) uses CPC only, not multiple SSL objectives; (c) no boundary refinement module; (d) no HAR downstream task.
- Classical change point detection (CUSUM, PELT, Bayesian online) — statistical, not representation-learning-based.
- **Multiple Change Point Detection in AR Time Series** (Ma et al., 2019, arXiv:1912.07775) — statistical, not deep learning.

### What's Still Missing
- **No work uses SSL prediction errors (from masked modeling or JEPA-style prediction) to discover activity boundaries in wearable/IMU sensor data.** TS-CP² uses contrastive separation, not prediction error.
- No boundary refinement module specifically designed for sensor-stream activity transitions (gradual vs. abrupt transitions differ from video).
- No unified framework combining multiple SSL pretext tasks where boundary detection is an emergent property of the SSL objective itself.

### PhD Contribution Potential: ✅ HIGH
BASS-HAR's core idea — that SSL prediction errors on continuous sensor streams *reveal activity boundaries* — is novel. TS-CP² uses contrastive embeddings for CPD but doesn't use prediction errors from masked modeling. The BRM concept is also unexplored for sensor data. This is a legitimate primary PhD contribution.

---

## AREA 2: Adaptive/Learned Segmentation for HAR (Replacing Fixed Windows)

### What's Been Done
- Virtually all HAR uses **fixed sliding windows** (e.g., 2s, 2.56s for WISDM/UCI-HAR). This is the dominant paradigm.
- **Sridhar & Myers (2021, arXiv:2112.12272)**: Propose a segmentation algorithm for wrist accelerometer data that identifies "salient activity segments" — but uses simple heuristics, not SSL.
- **OTAS** (Li et al., 2023, arXiv:2309.06276): Unsupervised boundary detection for temporal action segmentation — but **video only**, not sensors.
- **HOT-ROD** (Feng et al., 2023, arXiv:2307.04445): Discovers routines from continuous wearable data via Hawkes point process on time-series clusters — captures transitions but doesn't do HAR segmentation.
- Video domain has **TriDet** (Shi et al., 2023, arXiv:2303.07347): Relative boundary modeling with Trident-head — but for video, not sensors.

### What's Still Missing
- **No learned, adaptive segmentation for wearable HAR that replaces fixed windows.** The field is stuck on sliding windows.
- No method that jointly learns segmentation boundaries and activity recognition from continuous IMU streams.
- No analysis of how window boundaries affect HAR accuracy at transition zones (the "boundary problem" in HAR is acknowledged but not solved).
- No method that uses SSL prediction uncertainty to dynamically adjust window boundaries.

### PhD Contribution Potential: ✅ HIGH
Replacing fixed windows with SSL-discovered adaptive boundaries is a significant contribution. BASS-HAR's approach (prediction error → boundary → refined segmentation → classification) could eliminate the fixed-window paradigm for continuous HAR. This is the structural innovation of the framework.

---

## AREA 3: Cross-Domain Transfer in SSL for HAR

### What's Been Done
- **PRIMUS** (Das et al., 2024, arXiv:2411.15127): Pretraining IMU encoders with multimodal SSL (self-supervision + multimodal + nearest-neighbor). Evaluates on **out-of-domain datasets**. With <500 labeled samples, improves accuracy up to 15%. Key baseline for BASS-HAR.
- **TransfHAR** (Bradshaw et al., 2026, arXiv:2608.15861): Self-supervised wrist IMU pretraining on coarse activities, transfers to fine-grained activities absent from pretraining. Matches/exceeds supervised baselines by 6.2 balanced-accuracy points.
- **HALO** (Ding et al., 2026, arXiv:2608.27233): IMU foundation model with heterogeneity-aware SSL + language alignment. Handles sensing heterogeneity (sampling rates, channels, placements) and enables open-set recognition. 35M params, outperforms MOMENT (341M).
- **FedOpenHAR** (İşgüder & İncel, 2023, arXiv:2311.07765): Federated multi-task transfer learning across HAR datasets.
- **SAAC-JEPA** (Bouaziz et al., 2026, arXiv:2609.16071): Schema-adaptive JEPA for cross-machine transfer under partial sensor overlap — industrial, not HAR, but methodology is relevant.
- **Kaizen/CaSSLe** (Tang et al., 2024, arXiv:2401.02255): Continual SSL for wearable HAR — handles temporal domain shift across users/time.

### What's Still Missing
- **No work on cross-sensor-modality transfer for boundary detection.** Can SSL-discovered boundaries transfer from one sensor placement to another?
- No systematic study of how well SSL representations for boundary detection generalize across datasets (e.g., WISDM → UCI-HAR → PAMAP2 → USC-HAD).
- No work on whether boundary-aware SSL representations are more or less transferable than standard SSL representations.
- HALO addresses heterogeneity but doesn't consider boundary detection or continuous streams.

### PhD Contribution Potential: ✅ MEDIUM-HIGH
Could be a secondary contribution: show that BASS-HAR's boundary-aware representations transfer better across domains than standard SSL HAR representations. Particularly novel if boundaries transfer across sensor placements (wrist → waist → ankle).

---

## AREA 4: Label-Efficient HAR (Semi-Supervised, Few-Shot, Active Learning + SSL)

### What's Been Done
- **FedAR** (Presotto et al., 2021, arXiv:2104.08094): Combines semi-supervised + federated + active learning for HAR. Uses label propagation + active learning queries.
- **Consistency-based Weakly SSL** (Sheng & Huber, 2024, arXiv:2408.07282): Two-stage: SSL embedding + few-shot fine-tuning. Matches supervised performance on 3 datasets.
- **LLM-Guided Few-Shot HAR** (Ronando & Inoue, 2025, arXiv:2512.22385): LLM-generated knowledge priors for exemplar selection in few-shot wearable HAR. 88.78% macro-F1 on UCI-HAR.
- **PRIMUS** (above): <500 labels → 15% improvement.
- **TransfHAR** (above): 5 examples per class → 86.7% accuracy; 1-minute recording → 90.4%.
- **SelfMatch** (Kim et al., 2021, arXiv:2101.06480): Combines contrastive SSL + consistency for semi-supervised (not HAR-specific).

### What's Still Missing
- **No work combines boundary-aware SSL with label-efficient learning.** If boundaries are discovered without labels, the labeling effort shifts to *labeling segments* rather than *labeling windows*.
- No active learning strategy that queries labels at boundary regions (where activities transition) — these are the most informative and hardest examples.
- No semi-supervised method that uses SSL-discovered boundaries as structural priors for label propagation.

### PhD Contribution Potential: ✅ HIGH
BASS-HAR + label-efficient learning is a natural extension: boundary discovery enables segment-level labeling (1 label per segment vs. 1 label per window). Active learning at boundaries is novel. This could be a strong second paper.

---

## AREA 5: Boundary-Aware Contrastive Learning (Hard Negative Mining Across Boundaries)

### What's Been Done
- **Multimodal Contrastive HAR with Hard Negatives** (Choi et al., 2023, arXiv:2309.01262): Hard negative sampling for skeleton + IMU pairs. Uses adjustable concentration parameter. SOTA on UTD-MHAD and MMAct. **Closest to BASS-HAR's boundary-aware contrastive idea.**
- **Contrastive Learning with Hard Negatives** (Robinson et al., 2020, arXiv:2010.04592): Theoretical framework for hard negative sampling. General, not HAR-specific.
- **TimeHUT** (Jalali et al., 2025, arXiv:2510.01658): Hierarchical uniformity-tolerance balancing for time-series contrastive learning. SOTA on 128 UCR + 30 UEA datasets.
- **Good Contrastive Learning in TS** (Zhang et al., 2023, arXiv:2306.12086): Analyzes what makes good contrastive learning in time series.

### What's Still Missing
- **No work uses boundary information to define hard negatives in HAR.** The idea: windows that span a boundary (containing two activities) are natural hard negatives for both activities' pure windows.
- No contrastive framework where the SSL objective itself is boundary-aware — i.e., positive pairs respect boundary structure, negative pairs cross boundaries.
- No analysis of how boundary-aware negative mining affects representation quality at transition zones.

### PhD Contribution Potential: ✅ HIGH
This is a novel, well-defined contribution. The idea of using boundary-spanning windows as hard negatives is new. It directly extends BASS-HAR's architecture: the BRM discovers boundaries → contrastive loss uses boundary-aware sampling → better representations at transitions. Clean, publishable idea.

---

## AREA 6: Foundation Models / Pretrained Representations for HAR (2024–2026)

### What's Been Done
- **HALO** (Ding et al., 2026, arXiv:2608.27233): 35M-param IMU foundation model. Heterogeneity-aware SSL + language alignment. Open-set recognition. Trained on 10 HAR datasets, evaluated on 7 held-out. **Most relevant FM for BASS-HAR.**
- **PAT** (Ruan et al., 2024, arXiv:2411.15240): Foundation model for wearable movement data (actigraphy). Masked autoencoder pretraining on 21,538 participants. Mental health prediction focus.
- **PRIMUS** (Das et al., 2024, arXiv:2411.15127): IMU pretraining with multimodal SSL. Open-sourced. Evaluates cross-domain.
- **TransfHAR** (Bradshaw et al., 2026, arXiv:2608.15861): Wrist IMU SSL pretraining, real-time deployment.
- **HAR-JEPA** (Mohd Noor & Baraka, 2026, arXiv:2607.16350): JEPA for sensor HAR. VICReg regularization. Handles transitional activities (sit-to-stand, sit-to-lie). **Direct precursor to BASS-HAR.**
- **LaT-PFN** (Verdenius et al., 2024, arXiv:2405.10093): JEPA + PFN for zero-shot time-series forecasting.
- **MOMENT** (referenced in HALO): 341M-param general time-series FM.

### What's Still Missing
- **No foundation model for HAR that operates on continuous streams with boundary awareness.** All existing FMs use fixed-window inputs.
- No FM that discovers activity boundaries as part of pretraining (boundary-aware pretraining objective).
- No FM that handles the continuous-to-discrete mapping (stream → segments → labels) end-to-end.
- HAR-JEPA handles transitional activities but still uses window-based inputs — doesn't discover boundaries.

### PhD Contribution Potential: ✅ MEDIUM-HIGH (as a future extension)
BASS-HAR could be positioned as a "boundary-aware pretraining strategy" that future FMs adopt. The contribution is the pretraining objective (prediction error → boundary discovery), not necessarily a 350M-param model. This is more of a long-term direction than an immediate paper.

---

## AREA 7: Novel SSL Pretext Tasks for Continuous/Streaming Sensor Data

### What's Been Done
- **TimeMAE** (Cheng et al., 2023, arXiv:2303.00320): Masked modeling for time series with semantic units (non-overlapping sub-series). Decoupled visible/masked encoding. **Relevant to BASS-HAR's masked modeling component.**
- **MTS-DMAE** (Xu et al., 2025, arXiv:2509.16078): Dual-masked autoencoder for MTS. Two pretext tasks: reconstruct masked values + estimate latent representations. Feature-level alignment.
- **Self-Distilled TS** (Pieper et al., 2023, arXiv:2311.11335): data2vec for time series. Student-teacher scheme predicting latent representations from masked views. Non-contrastive.
- **HAR-JEPA** (arXiv:2607.16350): JEPA with local + long-term temporal modeling + improved VICReg.
- **Wearable SSL for Influenza Detection** (Kolbeinsson et al., 2021, arXiv:2112.13755): Studies SSL objective selection for wearable time-series. No principled selection method found.
- **SSL for Time Series Survey** (Zhang et al., 2023, arXiv:2306.10125): Comprehensive taxonomy: generative, contrastive, adversarial. 10 subcategories.

### What's Still Missing
- **No pretext task designed specifically to exploit activity boundaries.** Existing pretext tasks are domain-agnostic. A "continuity prediction" pretext task (predict whether two windows are from the same activity) is mentioned in BASS-HAR but hasn't been explored in literature.
- No pretext task that combines: (a) masked modeling for local feature learning, (b) prediction error for boundary discovery, (c) contrastive learning with boundary-aware sampling — all in one unified framework.
- No "transition prediction" pretext task: predict what activity comes next given the current activity (requires boundary awareness).
- No pretext task that treats the continuous stream as a **sequence of variable-length segments** rather than fixed windows.

### PhD Contribution Potential: ✅ HIGH
The "continuity prediction" pretext task in BASS-HAR is novel. Designing pretext tasks specifically for continuous sensor streams (not borrowed from vision/NLP) is a legitimate contribution. The key insight: prediction errors at boundaries are not noise but signal.

---

## CORE GAP ANALYSIS: SSL + Boundary Detection + Wearable Sensors

### The Gap Is Real and Significant

**No prior work combines all three:**
1. SSL (self-supervised learning) as the representation learning method
2. Boundary/transition detection in continuous sensor streams
3. Wearable/IMU sensors for HAR

**Closest existing work (each missing ≥1 element):**

| Paper | SSL? | Boundary Detection? | Wearable Sensors? | Notes |
|-------|------|---------------------|-------------------|-------|
| **TS-CP²** (2011.14097) | ✅ (CPC) | ✅ (CPD) | ❌ (generic TS) | Closest but no HAR, no IMU, no BRM |
| **OTAS** (2309.06276) | ✅ | ✅ | ❌ (video) | Video only, no wearable sensors |
| **SSCAP** (2105.14158) | ✅ | ✅ (indirect) | ❌ (video) | Video action segmentation |
| **HAR-JEPA** (2607.16350) | ✅ (JEPA) | ❌ | ✅ (IMU) | Handles transitions but no boundary detection |
| **PRIMUS** (2411.15127) | ✅ | ❌ | ✅ (IMU) | SSL pretraining, no boundaries |
| **TransfHAR** (2608.15861) | ✅ | ❌ | ✅ (wrist IMU) | SSL pretraining, no boundaries |
| **HALO** (2608.27233) | ✅ | ❌ | ✅ (IMU) | Foundation model, no boundaries |
| **Choi et al.** (2309.01262) | ✅ (contrastive) | ❌ | ✅ (skeleton+IMU) | Hard negatives, no boundaries |
| **Sheng & Huber** (2408.07282) | ✅ | ❌ | ✅ | Weakly SSL, no boundaries |
| **Sridhar & Myers** (2112.12272) | ✅ | Partial (heuristic) | ✅ (wrist accel) | Has segmentation algorithm but not SSL-based boundary detection |

**BASS-HAR occupies a genuinely novel position.** The specific combination of:
- Multiple SSL objectives (contrastive + continuity prediction + masked modeling) operating on continuous streams
- Prediction errors from SSL as boundary signals
- Boundary Refinement Module (BRM)
- IMU/wearable sensors for HAR

...has not been published. The gap is confirmed.

---

## SUMMARY: TOP 5 PhD-LEVEL RESEARCH IDEAS FOR BASS-HAR EXTENSION

### Idea 1 (Core): Boundary Discovery via SSL Prediction Errors on Continuous IMU Streams
- **Novelty:** TS-CP² uses contrastive separation; BASS-HAR uses prediction error from masked modeling + JEPA-style prediction. Different mechanism.
- **Contribution:** New SSL pretext task where boundary detection is an emergent property of prediction difficulty.
- **Feasibility:** High — builds directly on HAR-JEPA.
- **Paper target:** Top-tier venue (IEEE TMC, IMWUT, or NeurIPS/ICML workshop).

### Idea 2 (Extension): Boundary-Aware Contrastive Learning with Hard Negative Mining Across Transitions
- **Novelty:** Choi et al. (2309.01262) did hard negatives for HAR but not boundary-aware. BASS-HAR can define hard negatives as boundary-spanning windows.
- **Contribution:** New contrastive sampling strategy that respects activity boundaries.
- **Feasibility:** High — modular extension.
- **Paper target:** IEEE TMC, IMWUT, or AAAI.

### Idea 3 (Extension): Label-Efficient HAR via Boundary-Aware Active Learning
- **Novelty:** No work queries labels at boundary regions. BASS-HAR discovers boundaries → queries at boundaries → maximally informative labels.
- **Contribution:** Reduces labeling effort from window-level to segment-level; active learning at transitions.
- **Feasibility:** Medium-High.
- **Paper target:** IMWUT, IEEE TMC, or KDD.

### Idea 4 (Extension): Cross-Domain Transfer of Boundary-Aware SSL Representations
- **Novelty:** HALO and PRIMUS address cross-domain but not boundary-aware representations. Does boundary awareness help or hurt transfer?
- **Contribution:** Systematic study of transferability of boundary-aware SSL across sensor placements, datasets, and modalities.
- **Feasibility:** Medium — requires multiple datasets.
- **Paper target:** IEEE TMC, IMWUT, or CHIL.

### Idea 5 (Future): Boundary-Aware Pretraining for HAR Foundation Models
- **Novelty:** No FM for HAR incorporates boundary awareness. Current FMs (HALO, PAT, PRIMUS) use fixed windows.
- **Contribution:** Pretraining objective where boundary discovery is part of the SSL signal, enabling continuous-stream FMs.
- **Feasibility:** Lower — requires scale and compute.
- **Paper target:** NeurIPS, ICML, or ICLR (as a full paper).

---

## KEY REFERENCES (by relevance to BASS-HAR)

### Directly Related (SSL + HAR + Sensors)
1. **HAR-JEPA** — Mohd Noor & Baraka, 2026, arXiv:2607.16350 — JEPA for sensor HAR (supervisor's paper)
2. **PRIMUS** — Das et al., 2024, arXiv:2411.15127 — IMU SSL pretraining with cross-domain eval
3. **HALO** — Ding et al., 2026, arXiv:2608.27233 — IMU foundation model, heterogeneity-aware
4. **TransfHAR** — Bradshaw et al., 2026, arXiv:2608.15861 — Self-supervised wrist IMU, on-demand HAR
5. **Choi et al.** — 2023, arXiv:2309.01262 — Multimodal contrastive HAR with hard negatives
6. **Sheng & Huber** — 2024, arXiv:2408.07282 — Consistency-based weakly SSL for HAR
7. **Tang et al.** — 2024, arXiv:2401.02255 — Continual SSL for wearable HAR (Kaizen)
8. **Sridhar & Myers** — 2021, arXiv:2112.12272 — SSL + segmentation algorithm for wrist accelerometer

### SSL + Boundary/Change Point (Not HAR-specific)
9. **TS-CP²** — Deldari et al., 2020, arXiv:2011.14097 — **SSL change point detection with CPC** (closest to BASS-HAR concept)
10. **OTAS** — Li et al., 2023, arXiv:2309.06276 — Unsupervised boundary detection for video action segmentation
11. **SSCAP** — Wang et al., 2021, arXiv:2105.14158 — Self-supervised co-occurrence action parsing for video segmentation

### SSL Pretext Tasks for Time Series
12. **TimeMAE** — Cheng et al., 2023, arXiv:2303.00320 — Masked modeling with semantic units for TS
13. **MTS-DMAE** — Xu et al., 2025, arXiv:2509.16078 — Dual-masked autoencoder for MTS
14. **Self-Distilled TS** — Pieper et al., 2023, arXiv:2311.11335 — data2vec for time series
15. **TimeHUT** — Jalali et al., 2025, arXiv:2510.01658 — Hierarchical contrastive for TS
16. **SSL for TS Survey** — Zhang et al., 2023, arXiv:2306.10125 — Comprehensive taxonomy

### Foundation Models for Time Series / Sensors
17. **PAT** — Ruan et al., 2024, arXiv:2411.15240 — Foundation model for wearable movement data
18. **LaT-PFN** — Verdenius et al., 2024, arXiv:2405.10093 — JEPA + PFN for zero-shot TS forecasting
19. **SAAC-JEPA** — Bouaziz et al., 2026, arXiv:2609.16071 — Schema-adaptive JEPA for cross-machine transfer
20. **Clin-JEPA** — Yang et al., 2026, arXiv:2605.10840 — JEPA for EHR patient trajectories

### Label-Efficient HAR
21. **FedAR** — Presotto et al., 2021, arXiv:2104.08094 — Semi-supervised + federated + active learning for HAR
22. **LLM-Guided Few-Shot HAR** — Ronando & Inoue, 2025, arXiv:2512.22385 — LLM priors for few-shot exemplar selection
23. **FedOpenHAR** — İşgüder & İncel, 2023, arXiv:2311.07765 — Federated multi-task transfer for HAR

### Boundary Modeling in Video (Transferable Techniques)
24. **TriDet** — Shi et al., 2023, arXiv:2303.07347 — Relative boundary modeling for temporal action detection
25. **Joint SSL Video Alignment + Segmentation** — Shah Ali et al., 2025, arXiv:2503.16832 — Unified optimal transport for alignment + segmentation

### Behavioral Routines from Wearables
26. **HOT-ROD** — Feng et al., 2023, arXiv:2307.04445 — Routine discovery from unlabeled wearable data via Hawkes process
27. **Wearable SSL for Influenza** — Kolbeinsson et al., 2021, arXiv:2112.13755 — SSL objective selection for wearable TS

---

## COMPETITIVE LANDSCAPE NOTE

The field is moving fast (2024–2026):
- HALO (Aug 2026) and TransfHAR (Aug 2026) show IMU foundation models are emerging rapidly
- HAR-JEPA (Jul 2026) is already published by the supervisor — BASS-HAR extends this
- TS-CP² (2020) established the SSL→CPD link but never applied to HAR/wearables
- OTAS (2023) established SSL→boundary detection but only for video

**The window for BASS-HAR is now.** The gap between TS-CP² (SSL for CPD) and HAR (wearable sensors) has not been bridged. The longer this gap exists, the more likely someone else fills it. Recommend publishing the core framework paper promptly, then following with extensions (Ideas 2–4).
