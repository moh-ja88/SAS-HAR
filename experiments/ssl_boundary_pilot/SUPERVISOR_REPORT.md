# SSL Boundary Detection Pilot — Report for Supervision Meeting
**Student:** Mohammed Jasim (moh-ja88) · **Date:** 2026-09-19
**Scope:** Phase A de-risking of the BASS-HAR core hypothesis
**Artifacts:** `04_Experiments/Pilot_SSL_boundary/` (workspace) and
`experiments/ssl_boundary_pilot/` on the `phd-workspace` branch of SAS-HAR.

---

## 1. Executive summary

The pilot asked one question: **do self-supervised representations of wearable
IMU streams carry enough information to detect activity boundaries without
dense boundary labels?**

**Answer: yes — decisively — but the mechanism matters.** On a subject-disjoint
benchmark with exact ground truth, a linear probe on the frozen SSL encoder
achieves **boundary F1 = 0.72–0.77 (3 seeds), MAE ≈ 0.18 s**; an unsupervised
distributional contrast (MMD over 2 s embedding windows) reaches **4–4.7×
chance AUPRC** with **0.05 s localization**; and the proposal's own
continuity-prediction (CP) error carries a genuine moderate signal (F1 0.34).
Raw reconstruction error (MTM) and signal-derivative baselines carry none.

Two benchmark-construction problems were discovered and fixed along the way —
one about PAMAP2 itself (relevant to every future experiment in the thesis),
one about our own harness (caught by a falsification test now permanently in
the pipeline). Both are documented below because they change how we should
read any boundary-detection number in the literature that uses these datasets.

## 2. What was built (framework, reusable for the thesis)

| Component | File(s) | Notes |
|---|---|---|
| Continuous-stream PAMAP2 loader (100→50 Hz, 18 IMU channels, gap-safe chunking, run merging) | `pamap2_stream.py` | Handles the NULL-pause protocol structure explicitly |
| SSL encoder: 4-block depthwise-separable 1D-CNN (proposal §3.3-compatible) with **CP head** (JEPA-style stop-gradient latent prediction) + **MTM head** (30% contiguous-span masked reconstruction) | `ssl_model.py`, `train_ssl.py` | 30 epochs, AdamW+cosine, early stop on val subject |
| Concat benchmark with exact GT | `pilot_concat.py` | Real activity segments (2.5 s edge trim), shuffled, same-label adjacency avoided, GT = label-change cuts only |
| Evaluation protocol | `evaluate_boundary.py` | Tolerance-F1 (0.5 s, one-to-one greedy), AUPRC (chance-calibrated), localization MAE, NMS peak-picking, robust per-subject z-norm, thresholds selected on validation subject only |
| Evidence families | `exp2`–`exp4` | MMD-RBF (median heuristic, multi-scale), cut-supervised MLP head, probe-contrast (JS divergence of probe posteriors on ±windows) |

Design decisions frozen in `DESIGN.md` **before** runs, resolving the
proposal's open parameters (window 4 s / hop 1 s, 30% masking, tolerance 0.5 s,
normalization-only preprocessing per your directive, LOSO-lite splits
train {1,2,3,5,6,8} / val {4} / test {7}, S9 excluded — 95% NaN-gap recording).

## 3. Two discoveries about ground truth

**D1 — PAMAP2 protocol labels cannot yield sample-accurate boundary
boundaries.** The protocol sandwiches each activity between 30–250 s of
unlabeled NULL pause; the actual postural changes happen at unknown positions
inside those pauses. Empirically (subjects 4 & 7): motion peaks within ±0.5 s
of label edges: **0–4%**; correlation of motion energy with edge proximity is
**negative** at every window width up to ±10 s. Any method evaluated against
PAMAP2 label edges as boundary GT is measuring noise. This directly
substantiates the gap statement in our literature review (§7.2: a
boundary-annotated benchmark is itself a needed contribution) and motivates
the concat benchmark (standard CPD methodology, cf. CLaP/TSB-UAD evaluation).

**D2 — Our first concat benchmark was buggy, and the bug was instructive.**
The pieces were appended in chronological order (the intended shuffle was
never executed), so ~90% of "boundaries" were cuts *within* the same activity.
Against those, every evidence family — including a 94%-accurate supervised
probe — scored at chance, which is what first alerted us. The falsification
test (a method that *must* work detecting almost nothing) exposed it; the fix
(shuffle + adjacency handling) tripled the number of genuine transitions and
produced the results below. The probe check is now printed by every experiment
in the harness. *Lesson for the thesis: an upper-bound sanity probe belongs in
every evaluation pipeline before negative results are believed.*

## 4. Results (corrected benchmark; subject-disjoint; thresholds from val only)

| Evidence family | Supervision | F1 | AUPRC (chance ≈ 0.063) | MAE |
|---|---|---|---|---|
| Probe-contrast (JS of posteriors, ±0.25–2 s windows) | linear probe (activity labels) | **0.732 / 0.773 / 0.716** (seeds 42/43/44) | 0.37–0.43 | 0.17–0.20 s |
| Cut-supervised MLP head (two-sided embedding statistics) | cut positions only | 0.443 | **0.499** | 0.24 s |
| MMD-RBF, 2 s sides (unsupervised) | none | 0.292 (P .20 / R .51) | 0.256–0.284 | **0.05 s** |
| CP latent prediction error (proposal §3.5 head 2) | none | 0.337 (P .27 / R .46) | 0.112 | 0.22 s |
| MTM masked reconstruction error (proposal §3.5 head 3) | none | 0.124 | 0.064 | chance |
| Derivative baseline ‖xₜ−xₜ₋₁‖ | none | 0.088 | 0.059 | chance |

Encoder sanity throughout: linear probe **94.3% / 0.934 macro-F1**, 12
classes, **cross-subject** — the representation is strong; the variation above
is purely in how boundary evidence is extracted from it.

**Interpretation — the mechanism ladder:**
1. *Reconstruction error is dead.* Within-activity stochasticity dominates it;
   it will never separate transitions. Drop "errors as a free byproduct" as a
   claim.
2. *Prediction error (CP) is real but weak alone.* The proposal's CP-primary
   intuition survives, as a representation-shaping objective rather than a
   detector.
3. *Contrast across time at the right scale is the key operation.* MMD needs
   ≥2 s sides (0.5 s sides collapse to chance) — transitions are revealed by
   comparing distributions, not instantaneous errors.
4. *The supervision gap is large and that is the opportunity.* Same frozen
   features: unsupervised MMD AUPRC ≈ 0.28 vs cut-supervised head 0.50 vs
   probe-contrast 0.74 F1. Closing this gap **is** Phase B — feed confident
   unsupervised detections back as pseudo-labels (the proposal's iterative
   refinement loop), now with an empirical justification and a clear target.

## 5. Consequences for the proposal (specific revision requests)

| Proposal item | Pilot evidence | Suggested revision |
|---|---|---|
| §3.5 "CP error = primary boundary signal" | CP F1 0.34 alone | Boundary detection = **contrast/distributional module on SSL embeddings** (Phase A′); CP retained as representation objective |
| §3.5 MTM as secondary boundary signal | MTM at chance | MTM retained for representation only; remove boundary claim |
| §3.6 BRM with threshold τ_b | Evidence scales matter (2 s ≫ 0.5 s) | BRM should operate on **multi-scale two-sided contrasts**, not pointwise errors |
| §3.8 Phase B iterative refinement | 0.28 → 0.50 → 0.74 supervision ladder | Phase B = pseudo-label closure of the gap; pilot provides the harness to measure it |
| §4.2 PAMAP2 as primary boundary dataset | D1 | Use PAMAP2 with concat protocol, or switch boundary-GT primary to Opportunity locomotion / UCI-HAPT explicit transitions |
| SSL paradigm choice (contrastive vs JEPA, open per Revisions v4) | JEPA-lite CP already usable | Proceed with **JEPA-style** (supervisor's HAR-JEPA direction), now with pilot evidence it is compatible |

## 6. Honest limitations

- Concat cuts are *abrupt* distribution changes; natural transitions are
  gradual — Opportunity/HAPT validation (next step) still required before any
  external claim.
- Single test subject with usable GT (S7; S9 is 95% NaN) × 3 seeds. Full LOSO
  is cheap with the same harness and is queued.
- Chance-calibrated AUPRC ≠ SOTA comparison; SCOTT (~98% AUPRC, supervised
  online CPD) is a different task setup — comparison deferred to the paper.
- The supervised tiers of the ladder use activity labels / cut positions from
  *train subjects only*; no test leakage, but the label-free claim only
  applies to the MMD/CP tiers.

## 7. Next steps (proposed order, ~2–3 weeks)

1. Phase B loop v0: MMD peaks (≥2 s scale) → pseudo-cuts → retrain head →
   iterate; target: head F1 0.44 → ≥0.6 without new labels.
2. Head+probe ranking fusion (best AUPRC × best F1).
3. Opportunity locomotion natural-transition validation (GT changes
   mid-stream — no NULL-pause problem).
4. Full 9-fold LOSO + Wilcoxon significance for the paper-grade table.
5. JEPA encoder upgrade (EMA target network) — expected to lift the
   unsupervised tier.

## 8. Where everything lives

- Workspace: `PhDYasmeen/04_Experiments/Pilot_SSL_boundary/` — DESIGN.md
  (pre-registered), RESULTS.md (lab notebook incl. the Phase-1 correction),
  all runners + logs.
- Repo: `SAS-HAR` branch `phd-workspace`, directory
  `experiments/ssl_boundary_pilot/` (commit 7ad3e36).
- Reproduction: `python pilot_concat.py 42`, `exp2_distributional.py 42`,
  `exp3_trained_head.py 42`, `exp4_probe_contrast.py 42` (system Python 3.13,
  torch 2.6 CUDA; ~3 min/run).
