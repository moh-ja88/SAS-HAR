# Supervisor Meeting — Decision Memo
**Date prepared:** 2026-09-20 · **Basis:** pilot + Phase B v0 + 8-fold LOSO
(full data: `04_Experiments/Pilot_SSL_boundary/` — RESULTS.md, LOSO_RESULTS.md;
repo branch `phd-workspace`)

Three decisions are requested. Each has options, evidence, and a recommendation.

---

## Decision A — Proposal mechanism revision (§1.7, §3.5, §3.6, §3.8)

**Question:** adopt the pilot-verified mechanism (distributional contrast on SSL
embeddings) in place of "SSL prediction errors spike at transitions"?

**Evidence:**
- Naive error forms at/below chance across every test: MTM reconstruction
  AUPRC 0.069 (chance 0.062); derivative baseline 0.063 — LOSO-confirmed, 8 folds.
- Same frozen encoder: contrast mechanisms dominate with significance
  (Wilcoxon p = 0.0078, all folds): MMD 0.234, probe-contrast k=1 **0.379**.
- Encoder quality is not the bottleneck: 94% cross-subject linear probe.

**Options:** (1) revise now to contrast mechanism [recommended]; (2) keep
error-byproduct wording and risk the contradiction with our own pilot data.

**Recommendation:** Option 1. Wording drafted and ready; the gap statement,
objectives, and contributions remain unchanged — only the *mechanism* paragraph
family changes (C1 stays novel: same empty cell, stronger evidence).

## Decision B — Phase B direction (thesis core)

**Question:** invest in Phase B v1 (self-training with soft labels + joint
encoder fine-tuning) or pivot to the semi-supervised framing (few labeled
subjects + linear probe on SSL representations)?

**Evidence:**
- Phase B v0 (naive self-training): fails — student never exceeds teacher
  (AUPRC 0.256 → best 0.229, then decay); contraction + noise plateau measured.
  **LOSO corroboration:** the cut-supervised head also fails cross-subject
  (0.200 < MMD 0.234) — noisy pseudo-label training hurts transfer.
- LOSO label-fraction economics: **k=1 labeled subject → boundary AUPRC 0.379**
  (vs 0.234 unsupervised best, +62%); boundary evidence **saturates at k=1**
  (k-curve flat: 0.379/0.363/0.376/0.366) while classification accuracy scales
  (0.60→0.81). Labels calibrate the readout; the boundary structure already
  lives in the SSL representation.

**Options:** (1) Phase B v1 — 2–4 weeks, high risk/high purity (fully
label-free); (2) semi-supervised pivot — one labeled subject per deployment
as a calibration cost; the k-saturation curve is the headline figure;
(3) hybrid: pivot for P1, keep v1 as P2 stretch goal.

**Recommendation:** Option 3. P1 ships sooner with a defensible economics
story; the label-free question remains open as a contribution rather than a
dependency.

## Decision C — Boundary-GT dataset re-scope (§4.2)

**Question:** replace PAMAP2-as-primary with (a) concat-benchmark protocol
and/or (b) Opportunity/UCI-HAPT natural transitions?

**Evidence:** PAMAP2 protocol labels cannot yield sample-accurate boundary GT
(postural changes hide in 30–250 s NULL pauses; motion-peak alignment 0–4%).
Concat protocol gives exact GT but synthetic cuts; Opportunity locomotion
labels change mid-stream (natural); UCI-HAPT labels explicit postural
transitions (not yet on disk).

**Options:** (1) Opportunity-primary + PAMAP2-concat for controlled ablations
[recommended]; (2) PAMAP2-concat only (weaker external validity); (3) add
UCI-HAPT (small download, explicit transitions).

**Recommendation:** Option 1 (+3 if the download is approved). Data manifest
and loader plan are ready (`data/DATA_MANIFEST.md`).

---

## Attached evidence pack
1. `SUPERVISOR_REPORT.md` (+ Phase B addendum) — pilot narrative
2. `LOSO_RESULTS.md` — 8-fold tables, label-fraction curve, criteria verdicts
3. `RESEARCH_POLISH_PLAN.md` — completed/in-flight polish ledger
4. Open PR #1 (SAS-HAR) — loader fix, integrity cleanup, pilot code
