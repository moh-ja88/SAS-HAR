# Paper 1 — Skeleton (v0, 2026-09-20)
**Working titles (pick one):**
1. "How Self-Supervised Should Boundary Detection Be? Label Economics of
   Boundary Discovery in Continuous Wearable Activity Streams"
2. "From Prediction Errors to Predictive Contrasts: What Self-Supervised
   Representations Do and Don't Reveal About Activity Boundaries"
3. "One Labeled Subject Is Enough (Almost): Semi-Supervised Boundary Discovery
   from Self-Supervised IMU Representations"

**Venue shortlist (per proposal §5 + lit review):** IEEE Internet of Things
Journal · IEEE Trans. Instrumentation & Measurement · Pattern Recognition
(journal); ACM ISWC / IMWUT (conference alternative).
**Target:** ~14 pages, 8–9 tables/figures.

## Abstract (draft logic)
Fixed-window HAR assumes stationarity; boundary discovery in continuous IMU
needs no dense labels — or does it need any? We systematically compare five
evidence families extracted from a frozen self-supervised encoder (prediction
error, reconstruction error, unsupervised distributional contrast, pseudo-label
training, and light linear supervision) on exact-GT controlled benchmarks and
natural transitions. Findings: (i) error-byproduct evidence is a myth
(chance-level); (ii) distributional contrast at ≥2 s scales is the strongest
unsupervised signal but does not beat a derivative baseline on natural
transitions; (iii) a linear probe on the SAME representation, trained with a
single labeled subject, dominates everywhere — and boundary evidence
saturates at k=1 while classification keeps scaling; (iv) boundary-GT quality
(annotation lag, protocol NULL-structure) is a first-order confound we
quantify for two standard datasets, contributing an exact-GT concat protocol.

## 1. Introduction
- Fixed-window limitation → boundary-aware HAR; the label-cost question.
- Contribution list (target 4):
  C1 Evidence-family taxonomy + systematic comparison (first for wearable IMU).
  C2 The label-economics result: k-saturation curve for boundaries vs
     classification (Fig. 1 candidate).
  C3 GT-quality analysis: PAMAP2 NULL-pause invalidity + Opportunity lag
     quantification + exact-GT concat benchmark (released).
  C4 Practical recipe: SSL + one-session calibration probe ≥ everything
     unsupervised, across 2 datasets / 12 subjects / 2 tolerances.

## 2. Related work
Windowed/SSL HAR (TS-TCC family) · CPD (RuLSIF→CLaP, Bazarova SN-CPD) · video
action segmentation (OTAS, TCL) · SCOTT (supervised-contrastive CPD on HAR) ·
HAR-JEPA (supervisor's; windowed) · SWL-Adapt. Position: we are the first
controlled comparison of *where the boundary signal actually lives*.

## 3. Method (mostly done — reuse pilot §3 + parameterization)
Encoder (dw-sep CNN, CP/MTM heads) · evidence families §3a–3e (formulas
already in code+DESIGN docs) · evaluation protocol (tolerance-F1, AUPRC,
chance calibration, LOSO).

## 4. Experiments
- E1 Controlled: PAMAP2 concat (8-fold LOSO — table from LOSO_RESULTS §1–2).
- E2 Natural: Opportunity 4-fold (table from OPPORTUNITY_RESULTS).
- E3 Label economics: k-curve both datasets (headline figure).
- E4 GT-quality: alignment diagnostics (D1 + Opportunity lag plots).
- E5 Ablations (to run): MMD scale sweep; probe window scales; smoothing;
  encoder size; seed variance (3 seeds).

## 5. Discussion
- Why errors fail, contrasts work (RF blending + within-activity stochasticity).
- Why derivative is strong on natural transitions (motion bursts) — literature
  under-reports this baseline.
- Self-training failure modes (Phase B v0 + head transfer) as cautionary
  analysis for the pseudo-label path.
- Annotation lag as evaluation confound (tolerance choice matters: 0.5 s vs 2 s).

## 6. Limitations & future work
2 datasets; abrupt-cut benchmark; 4-subject Opportunity; no drill runs;
threshold transfer open; Phase B v1 (label-free closure) as future work.

## Status & gaps before drafting
| Piece | Status |
|---|---|
| All core numbers | ✅ pilot + LOSO + Opportunity (committed) |
| Headline figure (k-curve, both datasets) | ⬜ matplotlib script |
| Baselines: CLaP / PELT / RuLSIF runs | ⬜ required before submission (reviewers) |
| 3-seed stability for E1/E2 | ⬜ ~2 h compute |
| Scale ablations E5 | ⬜ ~1 day |
| Figures: evidence-curve examples, alignment plots | ⬜ |
| Writing | ⬜ after supervisor sign-off on framing (Decision B) |

**Framing dependency:** title/abstract assume the semi-supervised economics
framing (Decision B recommendation). If the supervisor prefers the fully
label-free framing, C2/C4 re-center on MMD + Phase B v1 instead.
