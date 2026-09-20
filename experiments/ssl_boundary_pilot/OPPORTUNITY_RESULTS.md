# Opportunity Natural-Transition Results — External-Validity Test
**Run:** 2026-09-20 · `opportunity_experiment.py` (design: OPPORTUNITY_DESIGN.md)
4-fold LOSO over S1–S4 (20 ADL runs, 6.5 h, 1,669 natural locomotion edges,
45 body-IMU channels @ 30 Hz, seam-masked). Tolerances: ±2 s (primary —
annotation lag measured at 64%/±2 s) and ±0.5 s (continuity with PAMAP2).

## Results (mean ± std of per-subject means; chance = near-edge fraction)

| Family | AUPRC @2 s | ×chance | AUPRC @0.5 s | ×chance |
|---|---|---|---|---|
| B0 derivative | 0.418 ± 0.052 | 1.24× | 0.145 ± 0.033 | 1.34× |
| CP prediction error | 0.357 ± 0.067 | 1.06× | 0.119 ± 0.026 | 1.10× |
| MTM reconstruction | 0.382 ± 0.065 | 1.14× | 0.152 ± 0.036 | 1.41× |
| MMD 2 s (unsupervised) | 0.425 ± 0.109 | 1.27× | 0.164 ± 0.045 | 1.52× |
| **Probe k=1** | **0.525 ± 0.095** | **1.56×** | **0.183 ± 0.045** | **1.69×** |
| **Probe k=3** | **0.549 ± 0.091** | **1.63×** | **0.190 ± 0.045** | **1.76×** |

Probe-contrast is the **best family in all 4 folds** at both tolerances;
k-curve rises monotonically (0.525→0.549).

## Pre-registered criteria verdicts (OPPORTUNITY_DESIGN.md §3)

| Criterion | Verdict | Detail |
|---|---|---|
| N1: MMD > 2× chance on ≥3/4 folds | **FAIL (as written)** | MMD is above chance in 4/4 folds but only 1.27×; "2× chance" was calibrated on sparse-GT concat (chance 0.06) — in this dense-GT regime (chance 0.34) multiplicative thresholds are scale-hostile. Criterion design flaw noted; not silently re-passed. |
| N2: probe k=1 ≥ MMD | **PASS** | 0.525 vs 0.425 (+10 pts), holds in 4/4 folds |
| N3: ordering probe>MMD>CP>MTM/B0 | **PARTIAL FAIL** | probe>MMD holds; but B0 (0.418) ties MMD (0.425) and CP is *worst* |

## The honest external-validity verdict

1. **The semi-supervised method transfers.** Probe-contrast on SSL
   representations is the top family in every fold, at both tolerances, on a
   different sensor suite (45 vs 18 ch), rate (30 vs 50 Hz), and transition
   regime (natural vs synthetic cuts). This is the P1 core claim — externally
   valid.
2. **The unsupervised tier does NOT clearly beat a derivative baseline on
   natural transitions** (MMD 0.425 vs B0 0.418 @2 s). On natural locomotion
   changes, raw motion energy is a strong baseline; representation-contrast's
   advantage (dramatic on synthetic cuts) largely evaporates. The paper must
   therefore carry B0 as an honest strong baseline, and the unsupervised
   tier should be positioned as analysis machinery, not headline performance.
3. **Annotation lag bounds achievable precision** at tight tolerance
   (@0.5 s everything compresses toward chance: best 0.190 vs 0.108). Dataset
   GT quality is a first-order factor in boundary evaluation — reinforces the
   benchmark-contribution argument (PAMAP2 finding + this lag analysis).
4. Label economics hold cross-dataset (k=1 → k=3 rising; one labeled subject
   already delivers most of the probe's value).

## Consequences for the P1 paper skeleton

- Headline method: **SSL representation + light-supervision boundary probe**
  (robust across both benchmarks).
- Unsupervised MMD: mechanism-analysis section (why contrast works) +
  PAMAP2-concat results where it shines; NOT the headline claim.
- Derivative baseline reported everywhere; its natural-transition strength
  is itself a finding (absent from much of the literature).
- Contribution list gains: **GT-quality analysis** (PAMAP2 NULL-pause
  invalidity + Opportunity lag quantification + concat protocol).

## Limitations

Drill runs excluded (scripted regime — future work); 4 subjects only (n=4
folds — Wilcoxon underpowered, not computed); single seed per fold; smoothing
bandwidth fixed at 0.1 s; threshold policy unchanged (AUPRC/oracle only for
probe).

## Artifacts

`results/opportunity/fold_s{1..4}.json` (per-run, both tolerances),
`opp_full.txt` run log. Reproduce: `python opportunity_experiment.py` (~10 min).
