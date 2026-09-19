# Results Directory

**No verified results documents live here.**

On 2026-09-19 the files `baseline_results.md`, `segmentation_results.md`, and
`ablation_studies.md` were removed. They were illustrative templates written
during project scaffolding, but their content presented concrete numbers
(e.g., B-F1 0.918, 97.2% accuracy, an RTX 3090 hardware table) as if they were
experimental results. No such experiments were run; the numbers are synthetic
and inconsistent with the actual experiment outputs.

## Where the real numbers are

- `experiments/BENCHMARK_RESULTS.md` — the only real SAS-HAR benchmark run
  (Opportunity, single seed; includes its own caveats)
- `experiments/all_datasets/all_results.json` — raw outputs (note: UCI-HAR /
  WISDM / PAMAP2 entries in this file are near-chance / broken runs)
- `experiments/ssl_boundary_pilot/RESULTS.md` — verified pilot results for the
  SSL boundary study (PAMAP2, subject-disjoint, seeds 42–44)

Do not reinstate results documents unless generated from actual experiment
artifacts with the runner, seed, and data version recorded.
