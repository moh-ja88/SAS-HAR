# Pilot Study Design: SSL Prediction Error as Boundary Signal
**Project:** BASS-HAR — Pilot SSL_boundary (Phase A de-risking)
**Hypothesis under test (H1):** Self-supervised prediction errors spike at activity
transitions in continuous wearable-IMU streams, enabling boundary detection without labels.
**Pre-registered success criteria** (defined before any run — see §7).

---

## 1. Purpose and scope

This pilot answers ONE question: *does the core BASS-HAR assumption hold?*
It is not a paper experiment. It deliberately excludes: classifier (Phase C),
boundary refinement loop (Phase B), label-efficiency claims, cross-dataset claims.

Scope decisions (resolving proposal ambiguities flagged in investigation):

| Ambiguity (source) | Pilot decision | Rationale |
|---|---|---|
| SSL paradigm: contrastive vs JEPA (Revisions v4 tension) | **Predictive latent error (CP/JEPA-style) = PRIMARY; masked reconstruction (MTM) = SECONDARY control** | Supervisor recommends JEPA-primary; CP head is JEPA-lite predictive coding and needs no negatives; MTM gives an interpretable visible-space control |
| ε_t computation (unspecified §3.5) | Per-channel MSE, mean over channels; reported raw AND smoothed (Gaussian σ=0.1 s) | Channel-comparable after z-score; smoothing is a reported hyperparameter, not a hidden choice |
| Window size / stride (unspecified §3.3) | Window 4 s (200 samples @ 50 Hz), hop 1 s (50 samples), 50% overlap aggregation | 4 s covers PAMAP2's shortest protocol activities; hop 1 s balances compute/resolution |
| Resolution & timestamp mapping (CNN stride 2 vs 0.5 s tolerance) | Total encoder downsampling = 4× (12.5 Hz frames, 80 ms); errors bilinearly upsampled to sample rate | 0.5 s tolerance = ±6 frames — comfortably above resolution |
| Preprocessing conflict (§3.2 Butterworth vs supervisor normalization-only) | **Normalization only** (per-channel z-score, train-subject statistics) + linear resample 100→50 Hz | Supervisor directive wins; resampling is a compute decision, not the contested band-pass |
| τ_b detection threshold (unspecified §3.6) | No single magic threshold: sweep thresholds on validation subject, report full P/R curves + F1 at best-val threshold, applied to test | Avoids threshold-fitting to test; honest reporting |
| Dataset (WISDM absent from proposal; v1.1 missing on disk) | **PAMAP2 Protocol** (primary, on disk); Opportunity as secondary sanity check if time permits | Proposal-primary; 100 Hz continuous; 9 subjects → LOSO-able; real transitions |

## 2. Data

- **Source:** `data/pamap2/PAMAP2_Dataset/Protocol/subject101..109.dat` (raw, on disk)
- **Channels:** 3 IMUs (hand/chest/ankle) × (temp + 6 inertial) = 21 → drop temperature → **18 channels**
  (cols: hand 4–9,15–20 → see `list_of_sensors.txt` mapping; HR excluded — many NaNs)
- **Stream construction:** per subject: load → drop NaN rows (ffill≤3 then drop, per sashar/pamap2.py
  convention) → resample 100→50 Hz (linear) → mark activity column → **transitions = label-change
  indices after merging segments < 1.0 s** (PAMAP2 protocol labels, incl. `activity0` transitions?
  No: activity0 = transient NULL → those spans are masked OUT of training, but label changes
  into/out of NULL are kept as boundaries only if both sides are valid activities; NULL spans
  themselves are excluded from windows)
- **Splits (LOSO-lite for pilot):** train SSL = subjects {1,2,3,5,6,8}; validation = {4};
  test = {7,9}. Full LOSO deferred to paper experiments. Subject-disjoint by construction.
- Normalization: per-channel mean/std computed on TRAIN subjects' pooled samples only.

## 3. Model (consistent with proposal §3.3 in spirit, pilot-scaled)

```
Input x ∈ [B, 18, 200]  (z-scored)
Encoder: 4× depthwise-separable Conv blocks (k=3), channels 32,64,128,128,
         stride 2,2,1,1  → total stride 4 → features [B,128,50]   (12.5 Hz)
Head A (CP / JEPA-lite, PRIMARY):
    predictor MLP on latent:  z_t = E(x)_t ; predict ẑ_{t+k} from z_t  (k = 4 frames = 0.32 s)
    loss = SmoothL1( pred(z_t), sg(E(x)_{t+k}) )   (stop-gradient on target = JEPA-style)
    boundary evidence: e^CP_t = ||pred(z_t) − sg(z_{t+k})||²  per frame, mean over dim
Head B (MTM, SECONDARY control):
    decoder: transposed-conv mirror of encoder → x̂ ∈ [B,18,200]
    mask: 3 contiguous spans, total 30% of window (proposal §3.5); mask embeds = 0-vector
    loss = MSE(x̂, x) on masked positions only
    boundary evidence: e^MTM_t = per-channel MSE at t, mean over channels (masked positions;
    unmasked positions excluded from evidence — they're trivially low-error)
Fusion (exploratory): ε_t = z-score(e^CP)_t + z-score(e^MTM)_t
```

## 4. Training protocol

- Optimizer AdamW lr 3e-4, wd 1e-4; cosine schedule; batch 64; epochs 30 (early-stop on val loss, patience 5)
- Augmentation: none in pilot (augmentations can suppress the very error signal we test)
- Seeds: 3 (42, 43, 44) — report mean ± std across seeds
- Hardware: system Python 3.13, torch 2.6.0+cu124 (CUDA if available, else CPU)
- Everything logged to `results/` as JSON + figures (per-subject error curves overlaid with GT transitions)

## 5. Evaluation protocol

Per held-out subject, per SSL head, per seed:

1. **Point-biserial effect size** (primary statistical test): correlation between binary
   is-transition (±0.5 s tolerance window) and ε_t across all samples. Wilcoxon rank-sum
   ε_t(near boundary) vs ε_t(≥1 s inside segments), p<0.05.
2. **Tolerance-F1** (detection metric): threshold ε at best-F1 on VALIDATION subject → apply to
   test subjects; peak-picking with NMS radius 0.5 s; TP if within 0.5 s of GT (matches
   `sashar/evaluation/metrics.compute_boundary_metrics`, tolerance=25 samples @ 50 Hz).
3. **Localization MAE** between matched detections and GT boundaries.
4. **Baselines (same data, same tolerance):**
   - B0 derivative: ||x_t − x_{t−1}||₂ threshold + NMS (non-learned)
   - B1 chance: random detections at matched density (10 runs, mean)
   - B2 supervised oracle ceiling (optional): logistic regression on one-hot label changes —
     sanity reference only, excluded from headline claims
5. **Consistency requirement:** per-subject results reported individually — no averaging that
   hides subject-level failure.

## 6. What the pilot does NOT claim

No SOTA comparison (SCOTT ~98% AUPRC is supervised-contrastive online CPD — different task setup;
positioned narratively only). No generalization beyond PAMAP2. No classifier. Numbers here are
pilot evidence for/against H1, not paper results.

## 7. Pre-registered success criteria (H1 supported iff ALL):

- **S1.** Point-biserial r > 0.20 AND Wilcoxon p < 0.05 on ≥ 6 of 8 SSL-head×test-subject×seed
  cells (2 heads × 2 test subjects × 2 of 3 seeds minimum per head).
- **S2.** Tolerance-F1 (primary CP head) exceeds derivative baseline B0 by ≥ +0.10 absolute F1
  on both test subjects (mean over seeds).
- **S3.** Localization MAE < 0.5 s (i.e., within tolerance) on average.
- **S4.** No head × subject cell where F1 < B1 chance + 0.05 (signal must not be chance-level).

If S1 fails → hypothesis rejected for this setup → pivot options (logged before running):
(i) latent-space JEPA with stronger predictor, (ii) multi-scale errors, (iii) revisit dataset choice.

## 8. Deliverables

```
04_Experiments/Pilot_SSL_boundary/
├── DESIGN.md            (this file — frozen before run)
├── pamap2_stream.py     continuous-stream loading + transition GT + splits
├── ssl_model.py         encoder + CP head + MTM head
├── train_ssl.py         pretraining loop
├── evaluate_boundary.py error curves → stats + tolerance-F1 + baselines
├── run_pilot.py         end-to-end driver (3 seeds)
└── results/             JSON + PNGs per seed/subject
```
