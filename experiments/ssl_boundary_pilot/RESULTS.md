# Pilot Results — SSL Boundary Evidence (H1 de-risking)
**Dates:** 2026-09-18/19 · **Framework:** this folder (see DESIGN.md)
**Benchmark (final, corrected):** PAMAP2 concat streams — real activity segments
(2.5 s edge trim), shuffled with same-label adjacency avoidance, exact cut GT at
label changes. SSL encoder trained on 6 train subjects (subject-disjoint),
val=S4, test=S7 (S9 excluded: 95% NaN-gap recording, no usable GT). Tolerance
0.5 s, NMS 0.5 s, thresholds from val subject only, robust per-subject z-norm.

## ⚠️ Phase 1 correction (2026-09-19)

The Phase-1 benchmark had a construction bug: pieces were appended in
chronological order (the documented shuffle was never executed), so ~90% of
"boundaries" were cuts WITHIN the same activity — no distribution change. All
Phase-1 negative verdicts were artifacts of this bug. It was exposed by a
falsification test (probe-contrast: only the ~11 genuine activity changes were
detected, 11/141) and fixed (shuffled pieces + same-label adjacency avoidance +
GT only at label changes). Lesson recorded: **always include an upper-bound
sanity probe before interpreting negative results.**

## Final results (corrected benchmark, subject-disjoint, val-selected thresholds)

| Evidence family | Supervision | F1 (S7) | AUPRC (chance≈0.063) | r | MAE |
|---|---|---|---|---|---|
| **Probe-contrast** (JS of probe probs, windows ±0.25–2 s) — seeds 42/43/44 | linear probe (labels) | **0.732 / 0.773 / 0.716** | 0.374 / 0.426 / 0.397 | +0.43 | 0.17–0.20 s |
| **Trained head** (MLP on two-sided embedding stats, cut-supervised) | cut labels only (no activity labels) | 0.443 | **0.499** | +0.37 | 0.24 s |
| **MMD-RBF** two-sided (side 2 s) — unsupervised | none | 0.292 (P .20 R .51) | 0.256–0.284 | +0.30–0.33 | 0.05 s |
| **CP** latent prediction error (JEPA-lite) | none | 0.337 (P .27 R .46) | 0.112 | +0.13 | 0.22 s |
| MTM masked reconstruction error | none | 0.124 | 0.064 | +0.00 | chance |
| B0 derivative baseline | none | 0.088 | 0.059 | −0.02 | chance |

Encoder sanity (same frozen weights throughout): linear probe **94% accuracy /
0.934 macro-F1** on pure windows, cross-subject, 12 classes.

## Verdict on H1

**H1 is CONFIRMED in refined form.** SSL representations of wearable IMU
streams carry strong boundary information — but the mechanism matters, in a
clean monotone ladder:

1. **Raw reconstruction error (MTM): no signal** — reconstruction is dominated
   by within-activity stochasticity. The proposal's "errors as free byproduct"
   reading is dead for reconstruction.
2. **Latent prediction error (CP): moderate genuine signal** (F1 0.34, AUPRC
   1.8× chance) — the proposal's CP-primary intuition is partially right.
3. **Unsupervised distributional contrast (MMD, 2 s sides): good** (AUPRC
   4–4.7× chance, MAE 0.05 s when detected) — scale matters: 2 s ≫ 0.5 s sides.
4. **Light supervision on the SAME frozen SSL features: dominant.**
   Cut-supervised head: AUPRC 0.50 (8× chance). Activity-probe contrast:
   F1≈0.74 stable across 3 seeds, MAE≈0.18 s, r≈0.43. The gap between (3) and
   (4) is the price of zero supervision; closing it is Phase B's job
   (self-labelling from confident detections, i.e. exactly the proposal's
   iterative refinement loop).

## The three permanent discoveries

- **D1 — PAMAP2 protocol labels cannot yield sample-accurate boundary GT**
  (postural changes hide inside 30–250 s NULL pauses; derivative peaks within
  ±0.5 s of label edges: 0–4%). Motivates a boundary-annotated benchmark as a
  contribution; concat protocol released in this folder.
- **D2 — Evidence-mechanism hierarchy**: representation-contrast ≫ pointwise
  errors. Boundary detection should be a first-class objective (trained head /
  distributional contrast), not an error byproduct.
- **D3 — Methodological**: benchmark bugs can masquerade as negative results;
  the probe-based falsification test that caught ours is now part of the
  harness (probe check printed by every experiment).

## Next steps (updated)

1. **Close the supervision gap**: feed confident MMD/probe detections back as
   pseudo-labels to retrain the head (Phase B loop, now empirically motivated).
2. **Head + probe fusion**: head has best AUPRC (0.50), probe-contrast best F1
   (0.74) — combine rankings.
3. **Multi-scale MMD** (2 s→4 s sides; gradual vs abrupt transition classes).
4. **Natural transitions**: Opportunity locomotion labels (change mid-stream,
   unlike PAMAP2 protocol) — the real-world test.
5. Full LOSO (9 folds) + significance tests (Wilcoxon) for the paper run.
6. JEPA encoder upgrade (current CP head is JEPA-lite; a full JEPA with
   EMA target may lift the unsupervised tier).

## Reproduction

```
python pilot_concat.py 42        # CP / MTM / B0 on corrected benchmark
python exp2_distributional.py 42 # MMD multi-scale
python exp3_trained_head.py 42   # cut-supervised head
python exp4_probe_contrast.py 42 # probe contrast (+ seeds 43 44)
python diagnose_signal.py 42     # D1 evidence (label-GT invalidity)
```
Logs: pilot_log*.txt, exp*_log.txt, re*_log.txt (2026-09-19 runs).
