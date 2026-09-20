# LOSO + Label-Fraction Experiment — Pre-Registered Design
**Frozen:** 2026-09-20, before implementation · **Runner:** `loso_experiment.py`
**Purpose:** convert the single-subject pilot numbers into subject-population
evidence before the supervisor meeting; quantify the semi-supervised option
(probe label-fraction curve) that Decision B (Phase B v1 vs pivot) hinges on.

---

## 1. Folds

- **Subjects 1–8** of PAMAP2 Protocol (S9/subject109 excluded: 95% NaN-gap
  recording — documented in pilot RESULTS.md).
- 8-fold leave-one-subject-out: each subject held out once; 7 train subjects.
- Per fold: SSL encoder retrained from scratch on train subjects' pure
  activity windows (same recipe as pilot: 30 epochs, early stop).
- Seed policy: `seed = 42 + fold_index` for SSL init, shuffles, and head
  training (avoids fold-correlated seeds). Single run per fold — subject
  variance dominates; seed averaging (3×8 folds ≈ 6 h) deferred to P1.

## 2. Evidence families evaluated per fold (on the held-out concat stream)

| Family | Threshold policy | Rationale |
|---|---|---|
| B0 derivative | none (AUPRC/oracle only) | paired baseline |
| CP prediction error | none (AUPRC/oracle only) | ladder completion |
| MTM reconstruction error | none (AUPRC/oracle only) | ladder completion |
| MMD 2 s sides | **label-free 2.5 z** (Phase B rule) + AUPRC/oracle | unsupervised tier |
| Trained head | **label-free p98.5** (Phase B rule) + AUPRC/oracle | pseudo-label tier |
| Probe-contrast, k ∈ {1, 2, 4, 7} labeled subjects | AUPRC/oracle only (see §4) | semi-supervised tier |

Concat stream per test subject: 1 shuffle, same builder as pilot
(`pilot_concat.build_concat`): 2.5 s edge trim, 4–30 s pieces, no same-label
adjacency, GT = label-change cuts. Same for the 7 train streams (used only for
head pseudo-cuts via MMD).

## 3. Threshold policy — honest limitation, stated up front

Threshold transfer across subjects is unsolved (pilot §limitations: val-swept
probe thresholds landed at 9–16 z robust — unstable). For LOSO we therefore
report:

- **AUPRC** (threshold-free ranking) as the **primary metric** for all families;
- **Oracle-F1** (best threshold swept on test GT) as a clearly-labeled **upper
  bound** — reporting only, never used by any method;
- Label-free fixed-threshold F1 **only** where Phase B already defined the rule
  (MMD 2.5 z; head p98.5), so the unsupervised tiers have an operating point.

For the probe label-fraction curve, AUPRC + oracle-F1 carry the story;
additionally the same k labeled subjects train a classification probe
(accuracy on held-out subject's pure windows) — the classic label-efficiency
curve — since the labels are legitimately available within budget k.

## 4. Label-fraction protocol (the Decision-B evidence)

For each fold and each k ∈ {1, 2, 4, 7} (subsets of the 7 train subjects,
rng-drawn):
- SSL encoder: ALWAYS trained on all 7 train subjects (unlabeled) — the
  semi-supervised separation: labels touch only the linear probe.
- Probe: logistic regression on k subjects' pure-window GAP embeddings.
- Boundary evidence: JS divergence between probe posteriors on left/right
  windows of the held-out stream (pilot exp4 mechanism, embeddings cached per
  fold — only the probe refits per k).
- Report: boundary AUPRC, oracle-F1, classification accuracy.

## 5. Statistics

- Per-family per-subject AUPRC/F1 tables; **mean ± std** across 8 folds.
- Paired **Wilcoxon signed-rank** (family vs B0 derivative, per-subject AUPRC,
  n = 8, α = 0.05, two-sided) — scipy.stats.wilcoxon.
- Chance level per stream (near-boundary fraction) reported for calibration.

## 6. Cost budget (measured from pilot timings)

Per fold ≈ 12–15 min: SSL ~1 min · streams+embeddings ~3 min · CP/MTM evidence
~3 min · MMD (test + 7 train streams) ~5 min · head train+evidence ~2 min ·
probe-contrast (cached pairs + 4 refits) ~1 min.
**Total ≈ 1.5–2 h**, background run, per-fold JSON written immediately
(crash-safe partial results).

## 7. Success criteria (set before running)

- **S1 (robustness):** probe-contrast (k=7) AUPRC > chance in ≥ 7 of 8 folds
  and mean AUPRC ≥ 2× chance.
- **S2 (supervision economics):** k=1 probe-contrast mean AUPRC ≥ MMD mean
  AUPRC (one labeled subject beats fully unsupervised) — the pivot argument.
- **S3 (ordering):** family ranking preserved: probe(k=7) > head > MMD > CP >
  MTM/B0 on mean AUPRC.
- Any criterion failing is reported as-is and feeds the memo with equal weight
  to success.

## 8. Deliverables

`results/loso/*.json` per fold → `LOSO_RESULTS.md` (tables, criteria verdicts,
memo-ready numbers) → `MEETING_DECISION_MEMO.md` updated with real curves.
