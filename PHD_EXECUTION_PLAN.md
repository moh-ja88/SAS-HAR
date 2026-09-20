# BASS-HAR PhD Execution Plan
## Phased Roadmap Connecting Research Ideas to Student's Work

**Student:** Mohammed Jasim (USM)
**Supervisor:** Dr. Mohd Halim Mohd Noor (HAR-JEPA, arXiv:2607.16350)
**Created:** Sep 17, 2026
**Status:** Draft for review

> **STATUS UPDATE (Sep 20, 2026) — Phase 1 COMPLETE, ahead of schedule and with a
> different verdict than this plan anticipated.** The pilot (2 days instead of
> 3–4 weeks) is in `04_Experiments/Pilot_SSL_boundary/` (+ repo `phd-workspace`
> branch). Key deviations from the Phase-1 design below: (a) naive "errors spike
> at transitions" is FALSIFIED for reconstruction (MTM = chance) — the signal
> lives in *distributional contrast on SSL embeddings* (probe-contrast F1 0.72–0.77;
> unsupervised MMD 4.5× chance); (b) PAMAP2 label edges are NOT valid boundary GT
> (NULL-pause structure) — a concat benchmark with exact GT was built instead;
> (c) naive Phase-B self-training fails (v0), making the Phase-2 loop design the
> central open question. Sections below retain their original (pre-pilot) text
> for the record; re-basing Phases 2–5 on pilot evidence is queued post-meeting.

---

## Overview

This plan translates the 5 PhD-level research ideas (in `02_Literature_Review/bass_har_research_ideas.md`) into a concrete, phased execution roadmap. Each phase builds directly on what the student has **already done** and produces a publishable artifact.

### What the Student Has Done So Far

| Work | Where | Status |
|---|---|---|
| WISDM baseline (window 50/100/150/200, best=200 @ 96.27%) | `04_Experiments/WISDM_Baseline/` | Done (has data leakage — fixed notebook created) |
| Explored CNN hyperparameters (feature maps, layers, kernel size) | In notebook | Done |
| Understood fixed-window problem (size + transition zones) | Written summary | Done |
| **Key intuition:** "compare adjacent segments to detect boundaries" | Written summary | Done — this IS the SSL continuity prediction concept |
| Literature on segmentation methods | `02_Literature_Review/` | In progress |
| Code infrastructure (PhD-HAR-Segmentation repo) | `05_Code/PhD-HAR-Segmentation/` | Has PAMAP2 + Opportunity + UCI-HAR data loaded |
| BASS-HAR proposal (SSL-first) | `01_Proposal/New_proposal.md` | Written, needs fixes |

### What's Ready to Use

- **Datasets already downloaded** in `05_Code/PhD-HAR-Segmentation/data/`:
  - PAMAP2 (9 subjects, 18 activities, 3 IMU placements) — **best for boundary experiments** (has transitions)
  - Opportunity (4 subjects, ADL+drill, 120+ sensors) — **complex, multi-sensor**
  - UCI-HAR (30 subjects, 6 activities, pre-windowed) — **no transitions** (fixed windows)
  - WISDM (36 subjects, 6 activities) — **student's baseline**
- **GPU:** RTX 3070 Ti (CUDA available)
- **Supervisor's code:** HAR-JEPA architecture (JEPA + VICReg for IMU)
- **Literature:** 79 papers reviewed (33 SSL+boundary, 46 SSL+HAR)

---

## Phase 1: De-Risk the Core Assumption (Pilot Experiment)
**Goal:** Prove that SSL prediction errors spike at activity transitions in IMU data
**Duration:** 3–4 weeks
**Pre-requisite:** Fixed WISDM notebook runs on Colab

### Why This Phase?

The entire BASS-HAR framework rests on one assumption:

> **SSL prediction errors increase at activity boundaries in wearable sensor data**

No one has demonstrated this for IMU. OTAS showed it in video. TS-CP² used contrastive separation (different mechanism). If this assumption fails, the framework collapses. **We must test it first.**

### Tasks

| # | Task | Details | Output |
|---|---|---|---|
| 1.1 | Run fixed WISDM notebook | Subject-wise split, normalization, F1/Confusion Matrix | Baseline accuracy ~80%, confusion matrix |
| 1.2 | Load PAMAP2 continuous stream | Use data in `05_Code/PhD-HAR-Segmentation/data/pamap2/`; PAMAP2 has frame-level labels → we know exactly where transitions are | Continuous IMU stream + ground-truth transition timestamps |
| 1.3 | Train simple SSL on PAMAP2 | Masked autoencoder (mask 30% of samples, predict masked values); 1D-CNN encoder; train on **all data** (no labels) | Trained encoder + per-timestep prediction error curve |
| 1.4 | Compare error curve vs ground-truth transitions | Overlay prediction error signal with labeled activity boundaries; compute correlation (Pearson/Spearman) | **Plot: prediction error vs transition locations** |
| 1.5 | Quantitative evaluation | Precision/Recall/F1 of boundary detection using simple thresholding on error curve | Boundary F1 score |

### Deliverable
- **Pilot report (2–3 pages):** "Do SSL prediction errors reveal activity boundaries in IMU data?"
- If YES → proceed to Phase 2
- If NO → investigate which SSL objective works (contrastive? JEPA? hybrid?)

### Connection to Student's Work
- Student already understands "adjacent segment comparison" — this pilot ** operationalizes his intuition** as an SSL pretext task
- Uses PAMAP2 data already in the repo
- Simple enough for student to implement independently

---

## Phase 2: Core Framework — BASS-HAR (Paper 1)
**Goal:** Build the full BASS-HAR framework and publish the core paper
**Duration:** 4–6 months
**Pre-requisite:** Phase 1 confirms the core assumption

### Paper Target
IEEE TMC / IMWUT / AAAI — "BASS-HAR: Boundary-Aware Self-Supervised Learning for Continuous Human Activity Recognition"

### Architecture (from proposal)

```
Continuous IMU Stream
      │
      ▼
┌─────────────┐
│ CNN Encoder │ ← 1D depthwise-separable conv
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│ Transformer     │ ← temporal context
└──────┬──────────┘
       │
       ▼
┌─────────────────────┐
│ SSL Pretext Heads   │ ← TCL + CP + MTM
│ (prediction errors  │
│  = boundary signal) │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ BRM                 │ ← Boundary Refinement Module
│ (denoises signal)   │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ Boundary-Aware      │ ← hard negatives = boundary-spanning windows
│ Contrastive Update  │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ Context-Aware       │ ← semi-supervised fine-tuning
│ Classifier          │
└─────────────────────┘
```

### Tasks

| # | Task | Details | Duration |
|---|---|---|---|
| 2.1 | Implement CNN Encoder | 1D depthwise-separable; 32→64 channels; based on HAR-JEPA encoder | 1 week |
| 2.2 | Implement Transformer | 2-layer, 4-head attention, positional encoding | 1 week |
| 2.3 | Implement 3 SSL heads | TCL (Temporal Contrastive Loss), CP (Continuity Prediction), MTM (Masked Time-series Modeling) | 2 weeks |
| 2.4 | Implement BRM | 1D conv + softmax; takes SSL error signal → sharp boundary probabilities | 1 week |
| 2.5 | Implement training loop | Phase A (SSL pretrain) → Phase B (boundary+SSL iterate ×3) → Phase C (fine-tune) | 2 weeks |
| 2.6 | Implement boundary-aware contrastive sampling | Use BRM boundaries to define hard negatives (boundary-spanning windows) | 1 week |
| 2.7 | Train on PAMAP2 | LOSO evaluation; 5%, 10%, 25%, 50%, 100% label ratios | 2 weeks |
| 2.8 | Train on Opportunity | Same protocol; more complex (multi-sensor) | 2 weeks |
| 2.9 | Baselines comparison | vs. HAR-JEPA, TS-TCC, SimCLR-HAR, fixed-window CNN (student's WISDM baseline) | 2 weeks |
| 2.10 | Ablation studies | (a) without BRM, (b) without boundary-aware contrastive, (c) each SSL head alone, (d) RuLSIF vs SSL for boundary | 2 weeks |
| 2.11 | Write paper | Following IEEE TMC template | 3 weeks |

### Evaluation Metrics
- **HAR:** Accuracy, Macro-F1, per-class F1, Confusion Matrix
- **Boundary:** Precision/Recall/F1 (tolerance ±0.5s, ±1.0s), Covering score, F-score
- **Label efficiency:** Accuracy vs. % labeled data (curves)
- **Comparison:** vs. HAR-JEPA, TS-TCC, SimCLR-HAR, supervised baselines

### Deliverable
- **Paper 1** submitted to IEEE TMC / IMWUT
- Reproducible code in `04_Experiments/PAMAP2/` and `04_Experiments/Opportunity/`

### Connection to Student's Work
- CNN encoder = extension of student's WISDM 1D-CNN (he already tuned feature maps, layers, kernel size)
- Student's WISDM baseline becomes **one of the comparison baselines**
- Student's "adjacent segment comparison" intuition = the CP (Continuity Prediction) head

---

## Phase 3: Label-Efficient HAR via Boundary-Aware Active Learning (Paper 2)
**Goal:** Show that boundary-aware SSL reduces labeling effort dramatically
**Duration:** 3–4 months
**Pre-requisite:** Phase 2 BASS-HAR works

### Paper Target
IMWUT / KDD / IEEE TMC — "How Many Labels Do We Need? Boundary-Aware Active Learning for Efficient HAR"

### Core Idea

Current HAR: label every window (thousands of labels).
BASS-HAR: SSL discovers boundaries → we have **segments** → label **1 per segment** (dozens of labels).

| Approach | Labels Needed (PAMAP2, ~1000 windows) | Accuracy |
|---|---|---|
| Fixed window + supervised | ~1000 | ~85% |
| Fixed window + 10% labels | ~100 | ~75% |
| **BASS-HAR + segment labeling** | **~50** | **~85%** |

### Tasks

| # | Task | Details |
|---|---|---|
| 3.1 | Implement segment-level labeling | After BASS-HAR discovers boundaries, aggregate predictions per segment; assign single label per segment |
| 3.2 | Implement active learning at boundaries | Query labels at transition zones first (most informative); compare random vs uncertainty vs boundary-based querying |
| 3.3 | Label efficiency curves | Accuracy vs. {5, 10, 25, 50, 100, 200, 500} labels on PAMAP2 + Opportunity |
| 3.4 | Compare labeling strategies | (a) random window labeling, (b) uncertainty-based, (c) boundary-aware segment labeling |
| 3.5 | Write paper | |

### Deliverable
- **Paper 2** submitted
- Shows BASS-HAR achieves ~85% accuracy with ~50 labels vs ~1000 for traditional approach

### Connection to Student's Work
- Student noted "activities differ in duration" → variable-length segments (not fixed windows)
- Student's observation about transition zones = the active learning query points

---

## Phase 4: Cross-Domain Transfer (Paper 3, Optional)
**Goal:** Show boundary-aware SSL representations transfer across datasets/sensors
**Duration:** 3–4 months
**Pre-requisite:** Phase 2

### Paper Target
IEEE TMC / CHIL — "Do Boundary-Aware Representations Transfer? A Cross-Domain Study"

### Research Question

> Does learning boundary-aware SSL representations make them **more or less** transferable across datasets, sensor placements, and modalities?

### Tasks

| # | Task | Details |
|---|---|---|
| 4.1 | Train BASS-HAR on PAMAP2 | Transfer encoder to UCI-HAR, WISDM, Opportunity |
| 4.2 | Train baseline SSL (no boundary awareness) on PAMAP2 | Same transfer |
| 4.3 | Compare transfer accuracy | Does boundary awareness help or hurt? |
| 4.4 | Cross-placement study | Wrist → waist → ankle (PAMAP2 has 3 placements) |
| 4.5 | Cross-dataset study | PAMAP2 → Opportunity → WISDM → UCI-HAR |
| 4.6 | Write paper | |

### Deliverable
- **Paper 3** (optional, depends on results)
- Addresses "generalization" datasets already in proposal

### Connection to Student's Work
- Student only tested on WISDM → this phase broadens to all 4 datasets (already downloaded)
- Tests whether the "boundary awareness" learned on one dataset transfers

---

## Phase 5: Foundation Model Direction (Long-Term)
**Goal:** Boundary-aware pretraining objective for IMU foundation models
**Duration:** 6–12 months
**Pre-requisite:** Phases 2+3, computational resources

### Paper Target
NeurIPS / ICML / ICLR — "Boundary-Aware Pretraining for IMU Foundation Models"

### Idea
- HALO (35M params) and PRIMUS use fixed windows
- BASS-HAR's boundary-aware SSL could be a **pretraining objective** for a foundation model
- The contribution is the **objective** (prediction error → boundary discovery), not a 350M-param model
- Could pretrain on PAMAP2 + Opportunity + WISDM + UCI-HAR combined, then zero-shot transfer

### Why Last?
- Requires more compute (multi-dataset pretraining)
- Requires Phases 2–3 to validate the approach first
- More ambitious — higher risk

### Connection to Student's Work
- Student's observation "more layers ≠ better" → suggests the bottleneck is training method, not model size
- Foundation models are the 2026 trend (HALO, PRIMUS, TransfHAR) — boundary-aware pretraining would be novel

---

## Timeline Summary

```
Phase 1: De-risk (Pilot)          ████ 3-4 weeks
Phase 2: Core Framework (Paper 1) ████████████ 4-6 months
Phase 3: Label-Efficient (Paper 2)████████ 3-4 months
Phase 4: Cross-Domain (Paper 3)   ████████ 3-4 months (optional)
Phase 5: Foundation Model         ████████████████ 6-12 months (long-term)

Total PhD output:
  - 3 confirmed papers (Phases 1-3)
  - 1 optional paper (Phase 4)
  - 1 ambitious paper (Phase 5)
  - = 4-5 publishable papers (sufficient for PhD)
```

---

## Immediate Next Steps (This Week)

1. **Student:** Run `04_Experiments/WISDM_Baseline/01_WISDM_Window50_FIXED.ipynb` on Colab
   - Upload WISDM raw data
   - Run all cells
   - Report: accuracy, F1, confusion matrix

2. **Student:** Read `02_Literature_Review/ssl_boundary_detection_har_literature_review.md`
   - Focus on §7.2 (SSL-first approach)
   - Extract comparison table for segmentation methods

3. **Hossam/Hermes:** Fix proposal
   - §3.2: normalization-only preprocessing (per supervisor)
   - References: audit for 2023+ compliance
   - Reconcile 4 contributions → 3 (match v4 revisions)

4. **Hermes:** Prepare Phase 1 pilot experiment notebook
   - Load PAMAP2 from `05_Code/PhD-HAR-Segmentation/data/pamap2/`
   - Simple masked autoencoder
   - Plot prediction error vs. ground-truth transitions

---

## Risk Assessment

| Risk | Impact | Mitigation |
|---|---|---|
| SSL errors don't spike at transitions (Phase 1 fails) | HIGH — framework collapses | Try different SSL objectives (contrastive, JEPA); fall back to RuLSIF statistical method |
| PAMAP2 boundaries too gradual (not sharp) | MEDIUM | Use Opportunity (more transitions); use tolerance window in evaluation |
| RTX 3070 Ti insufficient for large models | MEDIUM | Use Colab Pro (A100); keep models small (<10M params) |
| Supervisor wants changes to methodology | LOW-MEDIUM | Proposal already aligned with HAR-JEPA lineage; supervisor co-authored it |
| Someone publishes SSL+boundary+wearable before us | MEDIUM-HIGH | Gap confirmed unoccupied as of Sep 2026; speed matters — Phase 1 this month |

---

## Competitive Urgency

| Paper | Date | What they did | What they missed |
|---|---|---|---|
| TS-CP² | 2020 | SSL for change point detection | No HAR, no IMU, no BRM |
| OTAS | 2023 | SSL boundary detection | Video only |
| HAR-JEPA | Jul 2026 | JEPA for sensor HAR | No boundary detection |
| HALO | Aug 2026 | IMU foundation model | No boundaries, fixed windows |
| TransfHAR | Aug 2026 | Wrist IMU SSL | No boundaries |
| **BASS-HAR** | **Nov 2026?** | **SSL + boundary + wearable** | **← This is us** |

**The gap won't stay open. Phase 1 should start this month.**
