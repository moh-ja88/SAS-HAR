# LOSO Results — 8-Fold Leave-One-Subject-Out + Label-Fraction Economics
**Run:** 2026-09-20 · `loso_experiment.py` (design frozen in LOSO_DESIGN.md)
**Setup:** PAMAP2 subjects 1–8 (S9 excluded), SSL retrained per fold (seed 42+fold),
held-out subject → concat stream with exact GT (137–166 boundaries/fold,
mean chance AUPRC 0.062). AUPRC = primary (threshold-free); oracle-F1 =
test-swept upper bound (reporting only). Wilcoxon = paired vs B0, n = 8.

---

## 1. Family ladder (mean ± std over 8 held-out subjects)

| Family | AUPRC | ×chance | oracle-F1 | Wilcoxon p vs B0 |
|---|---|---|---|---|
| B0 derivative | 0.063 ± 0.004 | 1.0× | 0.096 | — |
| MTM reconstruction | 0.069 ± 0.005 | 1.1× | 0.129 | 0.0156 |
| CP prediction error | 0.131 ± 0.039 | 2.1× | 0.340 | **0.0078** |
| Trained head (pseudo-labels) | 0.200 ± 0.028 | 3.2× | 0.195 | **0.0078** |
| MMD 2 s (unsupervised) | **0.234 ± 0.038** | 3.8× | 0.337 | **0.0078** |
| **Probe-contrast k=1** | **0.379 ± 0.050** | **6.1×** | 0.599 | (all probe ks >chance in 8/8 folds) |

p = 0.0078 is the exact two-sided minimum for n = 8 (all folds favored) —
the CP/head/MMD advantages over B0 are significant despite the small n.

## 2. Label-fraction economics (Decision-B evidence)

| k labeled subjects | boundary AUPRC | boundary oracle-F1 | classification acc |
|---|---|---|---|
| 1 | **0.379 ± 0.050** | **0.599** | 0.599 |
| 2 | 0.363 ± 0.036 | 0.572 | 0.694 |
| 4 | 0.376 ± 0.038 | 0.585 | 0.751 |
| 7 | 0.366 ± 0.026 | 0.580 | 0.810 |

**Two headline patterns:**
1. **One labeled subject beats everything unsupervised** (0.379 vs MMD 0.234,
   +62%) — S2 criterion PASS. The semi-supervised economics are real at
   population level, not a single-subject artifact.
2. **Boundary evidence saturates at k = 1; classification keeps scaling**
   (acc 0.60→0.81). The k-curve is flat for boundaries. Interpretation: the
   SSL representation already contains the boundary structure; labels only
   calibrate the readout. One labeled session is the entire boundary-supervision
   budget this task needs — a sharper claim than the pivot memo assumed.

## 3. New findings vs the single-subject pilot

- **The trained head does NOT generalize** (0.200 < MMD 0.234 in LOSO;
  pilot had head 0.499 > MMD 0.256 on S7). Training on noisy MMD pseudo-labels
  overfits subject-specific evidence shapes. This *independently corroborates*
  the Phase B v0 failure — pseudo-label training without correction hurts
  cross-subject transfer. S3 strict-ordering criterion: FAIL for this reason.
- **Subject variance is material**: probe oracle-F1 spans ~0.45–0.72 across
  held-out subjects (pilot's S7 was an easy subject; mean oracle-F1 is 0.58).
  Honest paper numbers are the LOSO means, not the pilot's S7.
- Pilot ladder otherwise **confirmed with significance** (MTM/B0 dead; CP
  moderate; MMD best unsupervised; probe dominant).

## 4. Criteria verdicts (pre-registered, LOSO_DESIGN.md §7)

| Criterion | Verdict |
|---|---|
| S1 probe(k=7) robust: >chance in ≥7/8 folds AND mean ≥2× chance | **PASS** (8/8 folds; 5.9× chance) |
| S2 k=1 probe beats unsupervised MMD | **PASS** (0.379 vs 0.234) |
| S3 strict ordering probe>head>MMD>CP>MTM | **FAIL** (head < MMD in LOSO — see §3) |

## 5. Consequences for the meeting memo

- **Decision A (mechanism revision):** strengthened — ladder significant.
- **Decision B (Phase B v1 vs pivot):** sharpened — the k-saturation finding
  argues the semi-supervised framing is not a compromise but the *natural*
  reading: boundary supervision is a one-time calibration cost. Phase B v1
  (fully label-free closure) remains the P2 stretch goal.
- **Head demotion:** the cut-supervised head should be dropped from the
  framework's unsupervised tier (MMD is better AND simpler); revisit only
  with corrected pseudo-labels or joint encoder training.

## 6. Limitations (stated for the record)

- Single seed per fold (subject variance dominates; 3×8 seeds deferred to P1).
- Concat cuts are abrupt by construction; natural-transition validation
  (Opportunity/HAPT) still pending — this remains the largest external-validity
  gap (Decision C).
- Threshold transfer unsolved: fixed label-free thresholds exist only for
  MMD/head; probe rows are AUPRC/oracle only.
- S9 excluded (data quality); PAMAP2 protocol labels unusable as boundary GT
  (pilot D1) — concat benchmark is the stand-in.

## 7. Artifacts

`results/loso/fold_s{1..8}.json` (per-fold full metrics), `aggregate.json`,
run log `loso_full.txt`. Reproduce: `python loso_experiment.py` (~40 min, CUDA).
