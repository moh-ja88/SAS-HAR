# PhD Progress Log — Mohammed Jasim (BASS-HAR)

**Student:** Mohammed Jasim  
**Supervisor:** Dr. Mohd Halim Mohd Noor (USM)  
**Collaborator:** Abdulrahman Baraka (co-author HAR-JEPA, arXiv:2607.16350)  
**Topic:** Boundary-Aware Self-Supervised HAR using Wearable IMU Sensors  
**Folder:** `C:\Work\OpenCode\PhDYasmeen\`

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Timeline](#2-timeline)
3. [Methodology Evolution](#3-methodology-evolution)
4. [Current Proposal Structure](#4-current-proposal-structure)
5. [Literature Reviews](#5-literature-reviews)
6. [Baseline Experiments](#6-baseline-experiments)
7. [Supervisor Feedback Log](#7-supervisor-feedback-log)
8. [Files Inventory](#8-files-inventory)
9. [Open Items](#9-open-items)

---

## 1. Project Overview

**Framework name:** BASS-HAR (Boundary-Aware Self-Supervised HAR)

**Core idea:** SSL operates on continuous IMU stream FIRST → prediction errors reveal activity boundaries → BRM (Boundary Refinement Module) denoises them → feedback loop sharpens both representations and boundaries → classifier outputs activities.

**Research Gap:** No prior work uses SSL prediction errors to discover activity boundaries in wearable sensor data. OTAS demonstrated SSL-based boundary detection in video action segmentation only; adapting this to wearable sensors remains open.

**3 Objectives (per supervisor, Aug 2026):**
1. Boundary Detection — discover transitions using SSL prediction errors
2. Label-Efficient Representation Learning — SSL on unlabeled continuous data
3. Comprehensive Evaluation — multi-dataset, LOSO, comparisons with baselines

---

## 2. Timeline

| Date | Event |
|---|---|
| **Mar 2026** | Original HAR proposal (docx), early literature collection, novelty analysis |
| **Jul 13, 2026** | `New proposal.md` V3 written — BASS-HAR with BDN + SSL + bootstrap-and-refine |
| **Jul 26, 2026** | Supervisor meeting — 10 feedback points; lit reviews written; revisions doc created |
| **Aug 2, 2026** | Methodology switched from "Boundary-first" (RuLSIF→BDN→SSL) to "SSL-first" (SSL→prediction error→BRM) after user identified contradiction between lit review §7.2 and proposal §3.8 |
| **Aug 4, 2026** | Objectives reduced from 4 to 3 per supervisor spec; section numbering fixed to match figure |
| **Aug 10, 2026** | Methodology section ordering fixed (§3.3 CNN → §3.4 Transformer → §3.5 SSL → §3.6 BRM) to match architecture diagram |
| **Sep 17, 2026** | WISDM baseline notebook reviewed; data leakage identified (random segment split + overlap 50%); fixed notebook created with subject-wise split + normalization + validation + F1/confusion matrix |
| **Sep 18, 2026** | SSL boundary pilot framework built (`04_Experiments/Pilot_SSL_boundary/`): PAMAP2 continuous-stream loader, SSL encoder (CP JEPA-lite + MTM heads), pre-registered DESIGN.md. **Discovery D1: PAMAP2 protocol labels cannot provide sample-accurate boundary GT** (postural changes hide in 30–250 s NULL pauses; motion peaks within ±0.5 s of label edges: 0–4%) → concat benchmark with exact cut GT built |
| **Sep 19, 2026** | **Pilot completed — H1 confirmed in refined form.** Benchmark shuffle bug found via probe falsification test and fixed (Phase-1 negative results were artifacts). Final: probe-contrast on frozen SSL features F1 0.72–0.77 (3 seeds, MAE ~0.18 s); cut-supervised head AUPRC 0.50; unsupervised MMD 4.5× chance; CP F1 0.34; MTM/derivative at chance. Evidence-mechanism hierarchy established. SUPERVISOR_REPORT.md written; framework + results pushed to SAS-HAR `phd-workspace` branch |

---

## 3. Methodology Evolution

### Phase 1: Boundary-First (Jul 13 — Aug 2)
- RuLSIF (statistical change-point) detects boundaries FIRST
- BDN (Boundary Detection Network) learns from RuLSIF pseudo-labels
- SSL consumes boundaries as conditioning signal
- Training: bootstrap-and-refine (BDN init → SSL → BDN re-inference → repeat)

### Phase 2: SSL-First (Aug 2 — present) ← CURRENT
- SSL pretext heads operate on continuous stream FIRST (no boundaries needed)
- Prediction errors from Continuity Prediction (CP) and Masked Temporal Modeling (MTM) spike at activity transitions
- BRM (Boundary Refinement Module) denoises SSL prediction-error signal into sharp boundaries
- Discovered boundaries feed back: hard negatives for contrastive learning, aligned masks for MTM
- Training: Phase A (SSL pre-training) → Phase B (boundary discovery + iterative refinement, ~3 cycles) → Phase C (semi-supervised fine-tuning)
- RuLSIF demoted to optional fallback (ablation only)

**Why switched:** User noticed contradiction — lit review §7.2 proposed SSL-first (matching OTAS gap), but proposal §3.8 described boundary-first. SSL-first is stronger: (a) matches the discovered research gap, (b) supervisor published HAR-JEPA (interested in SSL as primary tool), (c) higher novelty.

---

## 4. Current Proposal Structure

**File:** `New proposal.md` (545 lines, 46KB)

| Section | Content | Status |
|---|---|---|
| §1.1 Background | HAR limitations (fixed windows, labels, transitions) | ✅ |
| §1.2 Problem Statement | 3 structural limitations, isolated study | ✅ |
| §1.3 Research Gaps | G1 (segmentation), G2 (label efficiency), G3 (transitional recognition) | ✅ |
| §1.4 Research Questions | RQ1–RQ4 | ✅ |
| §1.5 Objectives | 3 objectives (boundary detection, label-efficient SSL, evaluation) | ✅ |
| §1.6 Contributions | C1 (SSL boundary discovery), C2 (refinement regime), C3 (transitional classifier), C4 (unified architecture) | ✅ |
| §1.7 Scope & Assumptions | SSL prediction errors spike at transitions (key assumption, principal risk) | ✅ |
| §2 Related Work | Windowed HAR, adaptive segmentation, SSL for HAR, temporal action segmentation | ✅ |
| §3.1 Overview | Architecture diagram (SSL-first flow) | ✅ |
| §3.2 Stage 1 | Data acquisition & preprocessing | ⚠️ Still says band-pass Butterworth; supervisor says normalization only |
| §3.3 Stage 2 | CNN Encoder (depthwise-separable 1D-CNN) | ✅ |
| §3.4 Stage 3 | Transformer Encoder (temporal modeling) | ✅ |
| §3.5 Stage 4 | SSL Pretext Tasks (TCL, CP, MTM) — boundary discovery via prediction error | ✅ |
| §3.6 Stage 5 | Boundary Refinement Module (BRM) — denoises SSL signal | ✅ |
| §3.7 | Composite Objective (joint loss) | ✅ |
| §3.8 Stage 6 | Prediction-Error-Driven Training Schedule (Phase A/B/C) | ✅ |
| §3.9 Stage 7 | Context-Aware Transitional Classifier | ✅ |
| §4 | Evaluation Plan | ✅ |
| §5 | Timeline & Deliverables | ✅ |

---

## 5. Literature Reviews

### 5.1 SSL + Boundary Detection HAR Review
- **File:** `ssl_boundary_detection_har_literature_review.md` / `.pdf`
- **Papers:** 33 references (2023–2026)
- **Key finding:** No paper combines SSL + explicit boundary detection + HAR/IMU + continuous streams
- **Gap dimensions identified:** 3 (segmentation, label efficiency, transitional recognition)
- **Proposed framework (§7.2):** SSL-first — contrastive on continuous stream, temporal coherence positives, prediction errors for boundaries, iterative refinement
- **Note:** This lit review INSPIRED the SSL-first methodology switch

### 5.2 SSL for HAR Review
- **File:** `ssl_har_literature_review.md` / `.pdf`
- **Papers:** 46 references
- **Coverage:** Contrastive, predictive, masked modeling, JEPA for HAR

### 5.3 Literature Review Spreadsheet
- **File:** `BASS_HAR_Literature_Review_30_Papers.xlsx`
- **Content:** Comparison table of 30 papers (method, problem, results, limitations, future work)

---

## 6. Baseline Experiments

### 6.1 WISDM Fixed-Window CNN (Original — with data leakage)
- **Notebook:** `01_WISDM_Window50.ipynb` (received from student)
- **Setup:** WISDM v1.1, window=50, step=25, overlap=50%
- **Model:** 1D-CNN (Conv1d(3→32) → ReLU → MaxPool → Conv1d(32→64) → ReLU → GAP → Linear(64→6))
- **Training:** Adam lr=0.001, 10 epochs, batch=32
- **Split:** `train_test_split` on segments (random) — **DATA LEAKAGE**
- **Reported accuracy:** ~96% (inflated)
- **Issues identified:**
  1. Random segment split → overlap leakage (window N in train shares 50% data with window N+1 in test)
  2. No subject-wise split → model learns person patterns, not activity patterns
  3. No normalization
  4. No validation set
  5. No F1/Precision/Recall/Confusion Matrix
  6. Activity count error in report ("0 to 6" = 7, but code has 6 classes)

### 6.2 WISDM Fixed-Window CNN (Fixed — subject-wise split)
- **Notebook:** `01_WISDM_Window50_FIXED.ipynb`
- **Fixes applied:**
  1. Subject-wise split (80% train users / 10% val / 10% test)
  2. Normalization (StandardScaler fit on train only)
  3. Validation set (monitoring overfitting)
  4. Full evaluation suite (Accuracy + F1 + Precision + Recall + Confusion Matrix)
  5. Segmentation per group (train/val/test segmented separately)
- **Expected accuracy:** ~75–85% (realistic, vs 96% inflated)
- **Status:** Ready to run (student needs to upload WISDM data to Colab)

### 6.3 Window Size Experiments (student-reported)
- Student tried window sizes: 50, 100, 150, 200
- Best reported: 200 at 96.27% (but with same data leakage issue)
- These need re-running with subject-wise split

### 6.4 SSL Boundary Pilot — PAMAP2 (Sep 18–19, 2026) ← LATEST
- **Location:** `04_Experiments/Pilot_SSL_boundary/` (workspace) + `experiments/ssl_boundary_pilot/` (SAS-HAR repo, `phd-workspace` branch)
- **Purpose:** de-risk the core BASS-HAR assumption (SSL errors ↔ boundaries) before building the full architecture
- **Setup:** PAMAP2 Protocol, 50 Hz, 18 IMU channels, subject-disjoint (train {1,2,3,5,6,8} / val {4} / test {7}; S9 excluded — 95% NaN). SSL encoder: 4-block dw-separable CNN, CP head (JEPA-style latent prediction) + MTM head (30% span masking). Benchmark: real activity segments (2.5 s edge trim), shuffled, exact cut GT
- **Key findings:**
  - D1: PAMAP2 protocol labels cannot yield sample-accurate boundary GT (NULL-pause structure) — any boundary eval vs label edges is invalid
  - D2: evidence-mechanism hierarchy: probe-contrast (F1 0.72–0.77, 3 seeds) > cut-supervised head (AUPRC 0.50, F1 0.44) > unsupervised MMD 2 s (AUPRC 0.26–0.28, 4.5× chance, MAE 0.05 s) > CP error (F1 0.34) > MTM/derivative (chance)
  - D3: encoder sanity: linear probe 94% / macro-F1 0.934 cross-subject — representation is strong; boundary evidence extraction is the differentiator
  - Methodological: benchmark construction bug (unshuffled concat) exposed by falsification probe; probe check now permanent in harness
- **Consequence:** proposal §3.5 "CP error = primary boundary signal" must be revised to contrast/distributional mechanism on SSL embeddings (see SUPERVISOR_REPORT.md §5) — pending supervisor meeting
- **Reproduction:** `python pilot_concat.py 42`, `exp2_distributional.py 42`, `exp3_trained_head.py 42`, `exp4_probe_contrast.py 42` (~3 min each, CUDA)

---

## 7. Supervisor Feedback Log

**Meeting date:** July 26, 2026

| # | Feedback | Status |
|---|---|---|
| 1 | 3 objectives only, no SSL/technique names in objectives | ✅ Applied (Aug 4) |
| 2 | Pick one SSL paradigm (A: Contrastive / B: Predictive / C: JEPA) | ✅ Contrastive selected; JEPA noted as future |
| 3 | Preprocessing = normalization only (Stage 2) | ⚠️ NOT YET applied to proposal |
| 4 | References must be 2023+ | ⚠️ Audit not yet done |
| 5 | Requested literature review on SSL + boundary detection | ✅ Done (33 papers) |
| 6 | No "boundary aware" / "context aware" in general objective | ✅ Applied |
| 7 | General objective should be generic | ✅ Applied |
| 8 | Clarify SSL vs boundary detection relationship | ✅ Applied (SSL-first, feedback loop) |
| 9 | Reference update policy (2023+ only) | ⚠️ Listed in revisions doc, not yet audited in proposal |
| 10 | Contrastive-first scope | ✅ Applied |

---

## 8. Files Inventory

### Core Proposal Files
| File | Type | Size | Date | Description |
|---|---|---|---|---|
| `New proposal.md` | Markdown | 46KB | Aug 10 | BASS-HAR V3, SSL-first methodology |
| `New proposal.pdf` | PDF | 179KB | Aug 10 | Compiled proposal |
| `PROPOSAL_REVISIONS_v4.md` | Markdown | 10KB | Aug 4 | 10 supervisor feedback points + SSL/BDN relationship |
| `PROPOSAL_REVISIONS_v4.pdf` | PDF | 91KB | Aug 4 | Compiled revisions |

### Literature Reviews
| File | Type | Size | Date | Description |
|---|---|---|---|---|
| `ssl_boundary_detection_har_literature_review.md` | Markdown | 58KB | Jul 26 | 33 papers, gap analysis, SSL-first proposal §7.2 |
| `ssl_boundary_detection_har_literature_review.pdf` | PDF | 161KB | Jul 26 | Compiled |
| `ssl_har_literature_review.md` | Markdown | 33KB | Jul 26 | 46 papers on SSL for HAR |
| `ssl_har_literature_review.pdf` | PDF | 115KB | Jul 26 | Compiled |
| `BASS_HAR_Literature_Review_30_Papers.xlsx` | Excel | 13KB | Jul 26 | Comparison table |

### Baseline Experiments
| File | Type | Size | Date | Description |
|---|---|---|---|---|
| `01_WISDM_Window50.ipynb` | Notebook | ~9KB | Sep 17 | Original (with data leakage) |
| `01_WISDM_Window50_FIXED.ipynb` | Notebook | 20KB | Sep 17 | Fixed (subject-wise split, normalization, validation, full metrics) |

### Historical / Older Files
| File | Type | Description |
|---|---|---|
| `HAR Proposal 2025.docx` / `.pdf` | Original proposal (Mar 2026) |
| `har-proposal-complete-novelty-analysis.md` / `.pdf` | Novelty analysis (Mar 2026) |
| `har-proposal-novelty-analysis.md` | Shorter novelty analysis |
| `New proposal.docx` / `.txt` | Earlier export of BASS-HAR V3 |
| `README.md` | Original project readme |
| `Mohammed 1.docx` | Student's summary document |
| `Literature_Review_Contrastive_Predictive_ssl.docx` | Earlier lit review draft |

### Reference Papers (in folder)
| File | Description |
|---|---|
| `Noor et al. - 2017 - Adaptive sliding window...pdf` | Supervisor's earlier work on adaptive windows |
| `Deep similarity segmentation model for sensor-based.pdf` | Similarity-based segmentation |
| `Similarity_Segmentation_Approach...pdf` | Larger similarity segmentation paper |

### Code / Utilities
| File | Description |
|---|---|
| `convert_md_to_pdf_fpdf.py` | MD→PDF converter (fpdf) |
| `create_pdf_simple.py` | Simple PDF creator |
| `extract_comments.py` | DOCX comment extractor |
| `requirements.txt` | Python dependencies |

### Subdirectories
| Folder | Description |
|---|---|
| `adwin/` | ADWIN concept drift experiments |
| `OPTWIN/` | OPTWIN adaptive window experiments |
| `PhD-HAR-Segmentation/` | Segmentation experiments |
| `SWL-Adapt/` | Sliding window adaptation |
| `temp/`, `temp_har/`, `temp_pulmovec/` | Temporary experiment folders |

---

## 9. Open Items

### Immediate (proposal fixes)
- [ ] Fix §3.2 preprocessing: remove band-pass Butterworth, keep normalization only (supervisor feedback #3)
- [ ] Audit references in `New proposal.md` for 2023+ compliance (LeCun 2022, van den Oord 2018 flagged)
- [ ] Consistency pass on lit review PDFs vs SSL-first switch
- [ ] Post-meeting: apply evidence-mechanism revisions from SUPERVISOR_REPORT.md §5 (CP/MTM boundary claims → representation role; BRM multi-scale contrast)

### Experiments (post-pilot, priority order)
- [ ] Phase B loop v0: MMD peaks → pseudo-cuts → retrain head → iterate (close 0.28→0.74 supervision gap label-free)
- [ ] Head + probe ranking fusion
- [ ] Opportunity natural-transition validation (locomotion labels change mid-stream — no NULL-pause problem)
- [ ] Full 9-fold LOSO + Wilcoxon significance for paper-grade table
- [ ] JEPA encoder upgrade (EMA target) — lift unsupervised tier
- [ ] Run `01_WISDM_Window50_FIXED.ipynb` on Colab (superseded in priority by pilot, still useful as hygiene demo)
- [ ] Re-run window size experiments (50, 100, 150, 200) with subject-wise split (low priority)

### Research / Investigation
- [ ] Cross-sensor transfer potential
- [ ] Foundation model / pretrained representation potential
- [ ] Few-shot / active learning combined with SSL

### Infrastructure / hygiene
- [x] SSL boundary pilot pushed to SAS-HAR `phd-workspace` (Sep 19)
- [ ] Merge `phd-workspace` → `main` via PR (includes sashar/data loaders fix)
- [ ] Quarantine fabricated legacy results in `docs/results/` (template numbers presented as real)
- [x] Hermes gateway restored + auto-start on login (Sep 19)
