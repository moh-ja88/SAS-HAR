# BASS-HAR: Boundary-Aware Self-Supervised Human Activity Recognition

**PhD Research Project**

| | |
|---|---|
| **Student** | Mohammed Jasim |
| **Supervisor** | Dr. Mohd Halim Mohd Noor (USM) |
| **Collaborator** | Abdulrahman Baraka (co-author HAR-JEPA, arXiv:2607.16350) |
| **Domain** | Wearable IMU sensors, Self-Supervised Learning, Continuous HAR |
| **Started** | March 2026 |

---

## Project Summary

BASS-HAR is a framework that uses **Self-Supervised Learning (SSL) on continuous IMU streams** to discover activity boundaries without supervised labels or handcrafted heuristics. SSL representations separate activities sharply (pilot: 94% linear-probe accuracy, cross-subject); boundary evidence is extracted by **distributional contrast across time** on these representations (two-sided window comparison, trained heads, or light supervision), then refined into sharp boundaries that feed back to improve SSL training.

> **Pilot verdict (Sep 19, 2026):** hypothesis confirmed in refined form — see
> `04_Experiments/Pilot_SSL_boundary/RESULTS.md`. Naive pointwise prediction/reconstruction
> errors carry no boundary signal; contrast mechanisms on the same frozen encoder reach
> F1 0.72–0.77 (light supervision) and 4.5× chance AUPRC (unsupervised MMD). The
> 0.28→0.74 supervision gap defines the Phase B design space (naive self-training fails —
> see RESULTS.md §Phase B v0).

**Research Gap:** No prior work uses SSL — of any paradigm — to detect activity boundaries in continuous wearable IMU streams (nearest: CLaP on generic time series, SCOTT's supervised-contrastive CPD on HAR sensors, OTAS in video). Gap analysis: `02_Literature_Review/` (33+46 papers, critical addendum).

**3 Objectives:**
1. **Boundary Detection** — discover transitions via contrast evidence on SSL representations
2. **Label-Efficient Representation Learning** — SSL on unlabeled continuous data
3. **Comprehensive Evaluation** — multi-dataset, LOSO, comparisons with baselines

**Current status:** Phase A pilot complete (Sep 2026); framework revision pending
supervisor meeting. Workspace version-controlled (private); pilot code also in the
public SAS-HAR repo (`phd-workspace` branch).

---

## Folder Structure

```
PhDYasmeen/
│
├── README.md                          ← This file
├── PROGRESS.md                        ← Progress log (timeline, milestones, open items)
├── FOLDER_REORG_PLAN.md              ← Reorganization plan (reference)
│
├── 01_Proposal/                       ← Current proposal + supervisor revisions
│   ├── New_proposal.md                ← BASS-HAR V3 (SSL-first, Aug 2026)
│   ├── New_proposal.pdf
│   ├── PROPOSAL_REVISIONS_v4.md      ← 10 supervisor feedback points + SSL/BDN relationship
│   ├── PROPOSAL_REVISIONS_v4.pdf
│   └── archive/                       ← Older proposal versions (Mar–Jul 2026)
│
├── 02_Literature_Review/              ← Reviews, gap analysis, research ideas
│   ├── ssl_boundary_detection_har_literature_review.md     ← 33 papers, gap analysis
│   ├── ssl_boundary_detection_har_literature_review.pdf
│   ├── ssl_har_literature_review.md                         ← 46 papers on SSL for HAR
│   ├── ssl_har_literature_review.pdf
│   ├── bass_har_research_ideas.md                          ← 5 PhD extension ideas (27 refs)
│   ├── BASS_HAR_Literature_Review_30_Papers.xlsx           ← Comparison table
│   └── archive/
│
├── 03_Reference_Papers/               ← Key reference papers
│   ├── Noor_2017_Adaptive_sliding_window.pdf              ← Supervisor's earlier work
│   ├── Deep_similarity_segmentation_sensor.pdf
│   └── Similarity_Segmentation_Approach.pdf
│
├── 04_Experiments/                    ← All experiments
│   ├── WISDM_Baseline/
│   │   └── 01_WISDM_Window50_FIXED.ipynb                 ← Subject-wise split, normalized
│   ├── PAMAP2/                                            ← (planned)
│   ├── Opportunity/                                       ← (planned)
│   └── Pilot_SSL_boundary/                               ← (planned — de-risk core assumption)
│
├── 05_Code/                           ← All code
│   ├── PhD-HAR-Segmentation/          ← Segmentation experiments
│   ├── SWL-Adapt/                     ← Adaptive sliding window code
│   ├── adwin/                         ← ADWIN concept drift detection
│   ├── OPTWIN/                        ← OPTWIN adaptive window
│   └── utils/                        ← PDF conversion, comment extraction
│
├── 06_Notes/                          ← Student notes and paper summaries
│   ├── Mohammed_summary.docx
│   └── Enhanced_deep_learning_model_HAR.txt
│
└── _archive/                          ← Stale/legacy files (review later)
    ├── temp_har/                      ← HAR toolkit (Docker, run_*.py)
    ├── temp_pulmovec/                 ← Unknown
    ├── comments_docx.txt
    ├── env_text_docx.txt
    ├── requirements.txt
    └── README_old.md
```

---

## Key Documents

| What | Where | Description |
|---|---|---|
| Current proposal | `01_Proposal/New_proposal.md` | BASS-HAR V3, SSL-first methodology |
| Supervisor revisions | `01_Proposal/PROPOSAL_REVISIONS_v4.md` | 10 feedback points, SSL/BDN relationship |
| Literature review (SSL+BD) | `02_Literature_Review/ssl_boundary_detection_har_literature_review.md` | 33 papers, gap analysis |
| Literature review (SSL HAR) | `02_Literature_Review/ssl_har_literature_review.md` | 46 papers |
| Research ideas | `02_Literature_Review/bass_har_research_ideas.md` | 5 PhD extension ideas, 27 refs |
| Progress log | `PROGRESS.md` | Timeline, methodology evolution, open items |
| WISDM baseline (fixed) | `04_Experiments/WISDM_Baseline/01_WISDM_Window50_FIXED.ipynb` | Subject-wise split, full metrics |

---

## Methodology (SSL-First)

```
Phase A: SSL on continuous stream → prediction errors recorded
Phase B: prediction errors → BRM → boundaries → hard negatives → SSL refinement → repeat ~3×
Phase C: semi-supervised fine-tuning with classifier
```

**Components:**
1. CNN Encoder (depthwise-separable 1D-CNN)
2. Transformer Encoder (temporal modeling)
3. SSL Pretext Heads (TCL, CP, MTM) — prediction errors = boundary signal
4. Boundary Refinement Module (BRM) — denoises SSL signal
5. Context-Aware Classifier

---

## Open Items

- [ ] Fix §3.2 preprocessing (normalization only, per supervisor)
- [ ] Audit references for 2023+ compliance
- [ ] Run WISDM fixed notebook on Colab
-- [ ] Pilot experiment: SSL prediction errors vs labeled transitions (PAMAP2)
- [ ] Reconcile proposal contributions (4→3 per v4 revisions)

See `PROGRESS.md` for full list.
