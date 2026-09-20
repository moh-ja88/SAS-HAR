# Opportunity Natural-Transition Validation — Pre-Registered Design
**Frozen:** 2026-09-20 · **Runner:** `opportunity_experiment.py`
**Purpose:** external-validity test (Decision C): do the PAMAP2-concat findings
hold on NATURAL transitions in a different sensor suite, rate, and population?

## 1. Data investigation findings (2026-09-20)

- Raw on disk: `data/opportunity/raw/OpportunityUCIDataset/dataset/` —
  S1–S4 × ADL1–5 + Drill, 250 columns, 30 Hz.
- Authoritative layout (from dataset `doc/documentation.html`): label columns
  244–250 (1-based) → **locomotion = 0-based col 243** (values 1/2/4/5 + null 0);
  body IMUs = 0-based cols 37–101 in five 13-col blocks; we use the 9-channel
  subset per IMU (sashar convention): starts 37, 50, 63, 76, 89 → **45 channels**.
  (Earlier probe misread col 244 = HL_Activity; corrected.)
- **Alignment diagnostic (all 20 ADL runs, 1,669 valid–valid edges):**
  motion peaks within ±2 s of label edges **64%** (±0.5 s: 18%; median offset
  ~0.4 s). Verdict: unlike PAMAP2 (0–4%, structurally invalid), Opportunity
  locomotion GT is **valid with ~0.5–2 s annotation lag**.
- Processed npz on disk is windowed (23,603 × 110 × 60) — usable for probe
  sanity only; boundary work requires raw streams (this design).
- Drill runs excluded (scripted 10-s rotation — different transition regime);
  noted as follow-up.

## 2. Protocol

- **Streams:** per ADL run: 45 IMU channels, drop NaN rows, split at time gaps
  >0.1 s; per-subject concat of 5 runs in chronological order (gap-flagged).
  No resampling (native 30 Hz). Z-score with train-fold statistics only.
- **GT:** label-change edges where both sides are valid (non-null) labels,
  computed per chunk; null spans masked out of stats (as pilot).
- **Tolerance:** primary **±2 s** (=60 samples; justified by §1 lag), secondary
  ±0.5 s (=15) for continuity with the PAMAP2 numbers. NMS radius = tolerance.
- **Splits:** 4-fold LOSO over subjects S1–S4. Per fold: SSL trained from
  scratch on 3 subjects' pure windows (single-label runs ≥ window; window 4 s
  =120 samples @30 Hz, hop 1 s =30; seed 42+fold).
- **Families:** B0 derivative; CP; MTM; MMD 2 s sides (label-free 2.5 z);
  **probe-contrast** k ∈ {1, 2, 3} labeled subjects (AUPRC + oracle-F1 only —
  threshold policy unchanged from LOSO_DESIGN §3).
  **Head family DROPPED** — LOSO showed it fails cross-subject transfer
  (0.200 < MMD 0.234); rationale documented in LOSO_RESULTS §3.
- **Model:** same architecture, parameterized channels/window (45 ch, 120
  samples; stride-4 grid → 7.5 Hz frames; CP k=4 frames ≈ 0.53 s).
- **Statistics:** mean ± std across 4 folds; per-run tables; chance per stream.

## 3. Pre-registered success criteria

- **N1 (transfer):** MMD 2 s AUPRC > 2× chance on ≥3 of 4 folds.
- **N2 (economics):** probe k=1 AUPRC ≥ MMD AUPRC (one labeled subject beats
  unsupervised) on mean.
- **N3 (ordering):** probe(k=3) > MMD > CP > MTM/B0 on mean AUPRC.
- Failure on any is reported and carried to the memo unchanged.

## 4. Cost

Per fold ≈ 12 min (5 test runs × evidence curves + probe training on ~3
subjects' windows + MMD on 7.5 Hz grids). Total ≈ 50–60 min, crash-safe
per-fold JSON in `results/opportunity/`.
