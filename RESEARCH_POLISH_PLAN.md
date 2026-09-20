# Research Polish Plan — Findings & Actions
**Date:** 2026-09-19 · **Scope:** full audit of BASS-HAR research artifacts after the
SSL boundary pilot (evidence base: `04_Experiments/Pilot_SSL_boundary/RESULTS.md`,
`SUPERVISOR_REPORT.md`, proposal reference/baseline audit of 2026-09-19).

---

## Executive summary — ranked polish actions

| # | Action | Why it matters | Effort | When |
|---|---|---|---|---|
| 1 | **Revise proposal §1.7/§3.5/§3.6/§3.8 to the refined H1** (contrast-on-SSL-embeddings, not error-byproduct) | Current text states an assumption the pilot *falsified in its naive form*; a reviewer or the supervisor reading the proposal against the pilot report will spot the contradiction immediately | 2–3 h (with supervisor sign-off) | After meeting |
| 2 | **Reference overhaul: ~90% of 41 refs are pre-2023; zero 2025/26** | Direct violation of supervisor directive #4/#9; CLaP/SCOTT/Bazarova/HAR-JEPA/Koopman-JEPA are *already cited in the lit reviews* but never imported into the proposal | 3–4 h | Pre-meeting (mechanical) |
| 3 | **Add missing baselines to §4.4**: CLaP, SCOTT, OTAS-adapted, SWL-Adapt, PELT/BOCPD | These are the closest-prior-work methods a reviewer will demand; SWL-Adapt is the supervisor's own — its absence looks like avoidance | 1 h (table) + implementation later | Table pre-meeting |
| 4 | **P1 empirical ladder**: LOSO, natural transitions (Opportunity/HAPT), 2–3 baselines actually run, Wilcoxon | Pilot is 1 test subject × 3 seeds on synthetic cuts — not publishable alone; the ladder converts pilot → Paper 1 | 2–4 weeks | Now |
| 5 | **Decide Phase B v1 vs semi-supervised pivot** | Phase B v0 failed; the 0.28→0.74 supervision gap is the thesis's central economic question | meeting decision | Meeting |
| 6 | **Put the umbrella workspace under version control** (private repo) | Proposals, lit reviews, PROGRESS.md, pilot = single copy on one disk; already lost-work once (sashar/data loaders) | 30 min | This week |
| 7 | Mechanical fixes: cross-refs, orphan citation, subsection labels | Credibility details | 15 min | Done (see D) |

---

## A. Proposal ↔ pilot consistency matrix (evidence-driven revisions)

The pilot produced verified evidence that specific proposal claims need revision.
Each row: current text → evidence → required revision.

| § | Current claim (quoted/summarized) | Pilot evidence | Required revision |
|---|---|---|---|
| §1.7 | "Key assumption: the SSL prediction errors spike reliably at activity transitions… principal technical risk" | MTM reconstruction = chance; CP error F1 0.34; contrast mechanisms F1 0.29–0.77 | Replace with: *SSL representations separate activities sharply (94% linear probe); boundary evidence = distributional contrast across time at ≥2 s scales, optionally supervised by few labels* |
| §3.5 | CP error = primary boundary signal; MTM = secondary | CP moderate, MTM dead; MMD/head/probe dominant | Recast CP/MTM as **representation objectives**; boundary detection = dedicated contrast module (Phase A′) |
| §3.6 | BRM denoises pointwise error signal, NMS above τ_b | Scale is the operative variable (2 s ≫ 0.5 s sides); pointwise errors carry no signal | BRM operates on **multi-scale two-sided contrasts** (0.5–4 s), not pointwise error |
| §3.8 | Phase B: I≈3 iterations of discover→mine→retrain; "prediction errors recorded but NOT yet used" in Phase A | Phase B v0 (naive) **contracts**: teacher AUPRC 0.256 vs student best 0.229; three measured failure modes | Phase B v1 requirements: soft/confidence-weighted labels, calibrated detection budgets, **joint encoder fine-tuning** — or semi-supervised pivot (few labeled subjects + linear probe, F1 0.74 already demonstrated) |
| §4.2 | PAMAP2 primary boundary dataset | D1: PAMAP2 protocol labels cannot yield sample-accurate boundary GT (NULL-pause structure; 0–4% motion-peak alignment) | Either (a) concat-benchmark protocol as we built, or (b) **Opportunity locomotion / UCI-HAPT transitions as natural-GT primary**; PAMAP2 demoted to representation/label-efficiency dataset |
| §3.2 | (fixed 2026-09-19) Butterworth removed, normalization-only | Supervisor directive #3 | ✅ done |
| §4.7 ref | "testable (§4.7)" | wrong target | ✅ fixed → §4.6 |

## B. Reference audit (supervisor directive #4/#9)

- 41 references: **49% pre-2020, 41% 2020–22, 3 refs ≥2023, zero 2025/26**.
- **Ready to import from lit reviews** (citations already written there):
  CLaP (Ermshaus/Schäfer/Leser 2025, arXiv:2504.01783) · SCOTT (Liu et al. 2024,
  arXiv:2403.10787) · Bazarova et al. 2024 (spectral-norm SSL-CPD) · HAR-JEPA
  (Noor & Baraka 2026, arXiv:2607.16350 — supervisor's own) · Koopman-JEPA
  (Ruiz-Morales et al. 2025) · OTAS (Li et al. 2023 — currently an orphan).
- **Keep as foundational** (defensible): datasets (Roggen 2010, Reiss 2012,
  Anguita 2013, Reyes-Ortiz 2016, …), Transformer (Vaswani 2017), RuLSIF
  (Liu 2013), DeepConvLSTM (Ordóñez 2016), BYOL/SimCLR/MAE (method origins).
- **Replace or drop**: mid-tier 2019–22 method citations whose role is
  "recent related work" (they no longer are): Farha 2019, Saeed 2019/2020/21/22
  cluster, Haresamudram 2020/21/22, Zerveas 2021, de Sá 2021, Minh 2021,
  Erdogan 2022, Deldari 2022, Yang 2022 — keep only those actually used as
  baselines in §4.4, reclassify the rest, and add 2024–26 successors.
- Formatting: convert stray "[Li et al., 2023]" to numeric style; fix dual-year
  ref [20]; regenerate PDF after edits (`_archive/convert_md_to_pdf_fpdf.py`).

## C. Baseline gaps in §4.4 (P1 defense surface)

| Missing baseline | Why reviewers demand it | Where cited already |
|---|---|---|
| CLaP (2025) | Self-supervised state detection — "conceptually identical" task on generic TS | lit review Addendum B |
| SCOTT (2024) | ~98% AUPRC online CPD **on HAR data** — the number to beat/contextualize | lit review Addendum D |
| OTAS-adapted (2023) | The video method our approach transfers — the "ported" criticism must be pre-empted by actually comparing | lit review §5.3 |
| SWL-Adapt (supervisor) | Adaptive sliding window; sharp empirical contrast also demonstrates distance from supervisor's own line | supervisor's repo on disk |
| PELT / BOCPD | Classical CPD floor beyond RuLSIF | standard |
| Fixed-window SSL (TS-TCC etc.) | Already listed ✅ | §4.4 |

## D. Mechanical defects (status)

- ✅ §1.7 cross-ref §4.7→§4.6 (fixed 2026-09-19)
- ✅ §3.5 "BRM (§3.3)"→§3.6 (fixed 2026-09-19)
- ⬜ §3.5 subsection numbering (5.1–5.3 mislabels) — fix with next content edit
- ⬜ Orphan OTAS citation → add to reference list (do with reference overhaul)

## E. P1 empirical ladder (pilot → publishable Paper 1)

1. **Full LOSO** (9 folds) of the 3 winning families (MMD / head / probe-contrast)
   + Wilcoxon vs baselines — harness exists, ~1 day compute.
2. **Natural-transition validation**: Opportunity locomotion labels (change
   mid-stream; 30 Hz; processed npz on disk) and UCI-HAPT explicit postural
   transitions. This is the credibility step beyond synthetic cuts.
3. **Run 2–3 real baselines**: CLaP (code public), RuLSIF/PELT (ruptures pkg),
   derivative+SWL-Adapt-style adaptive window. SCOTT if code available.
4. **Ablations**: MMD scale sweep (0.5–4 s), feature-set ablation, probe
   label-fraction curve (1/2/4 labeled subjects) — turns the supervision-gap
   finding into a headline figure.
5. **Seed protocol**: 3 seeds minimum, report mean±std (already in place).

## F. Positioning polish (narrative)

- New headline story (pilot-verified): *"SSL representations of IMU streams
  separate activities almost perfectly; boundary discovery is a
  representation-contrast problem, not an error-byproduct problem; the
  supervision gap (0.28→0.74 AUPRC→F1) is the design space."*
- Differentiation sentences needed vs: SWL-Adapt (adaptive window ≠ learned
  boundary evidence), HAR-JEPA (classification, not segmentation), CLaP
  (generic TS, no IMU/multi-scale gradual transitions), OTAS (video, dense
  features vs sparse kinematics).
- Contribution C1 wording must change from "SSL prediction errors reveal
  boundaries" to the contrast formulation (same novelty cell, stronger
  evidence, survives review).

## G. Infrastructure hygiene

- ⬜ **Version-control the umbrella workspace** (private GitHub repo; exclude
  `05_Code` nested clone via .gitignore, exclude `_archive` bulk or use LFS
  decision). Everything except the SAS-HAR clone is currently unversioned.
- ⬜ Delete junk: `nul` files (root + pilot folder), `temp*` folders are
  already archived; pilot `*_log.txt` are intentional (lab notebook) — keep.
- ⬜ WISDM fixed notebook still Colab-path-dependent; optional local rerun
  with WISDM v2.0 on disk (nice-to-have only).
- ✅ Proposal §3.2 + cross-refs fixed; PDF regeneration pending next edit batch.

## Sequencing

**Pre-meeting (this week):** ✅ #2 reference import (refs [42]–[48] added 2026-09-19:
HAR-JEPA, CLaP, Koopman-JEPA, SCOTT, OTAS, Bazarova, SWL-Adapt; 18 in-text citations
consistent; disclaimer updated) · ✅ #3 baseline table rows added (§4.4 boundary/CPD +
adaptive-windowing families; §2.5 nearest-prior-work expanded; §2.3 JEPA paragraph) ·
✅ #6 workspace git init (local commit 1749fd1, 65 files; remote push pending account
decision) · ✅ proposal PDF regenerated via pandoc/xelatex.
**Meeting decisions:** #1 substantive proposal revisions, #5 Phase B v1 vs
semi-supervised pivot, Opportunity-primary dataset switch (§4.2).
**Post-meeting:** #4 P1 ladder execution (weeks 1–4), concurrent P1 draft.

---

# Round 2 Findings (2026-09-20) — deeper audit after Round-1 mechanical fixes

## A. Stale-document cascade — four documents still assert the pre-pilot world

All four predate the pilot and none reference its findings. A reader (or the
supervisor) entering from any of them gets the *falsified* naive mechanism:

| Document | Problem | Status |
|---|---|---|
| `README.md` (workspace root) | Summary + objectives + gap all state "SSL prediction errors spike at transitions" — the naive form the pilot falsified (MTM=chance); no mention of pilot/refined mechanism | ✅ updated 2026-09-20 (see below) |
| `PHD_EXECUTION_PLAN.md` (Sep 17) | Phase 1 (pilot, 3–4 wks) is **already done — in 2 days — with a different verdict than planned**; phases 2–5 assume error-byproduct mechanism; "prerequisite: WISDM notebook on Colab" obsolete | ✅ status block added; full rebasing deferred to post-meeting (framing depends on Phase B decision) |
| `02_Literature_Review/bass_har_research_ideas.md` | 5 research areas with pre-pilot potential ratings; Area 1 (boundary detection) now has empirical evidence and a partially negative sub-result (self-training) — unrated/unranked | ⬜ addendum recommended post-meeting |
| `05_Code/.../docs/paper_outlines.md` (legacy repo) | Three paper outlines for the ABANDONED SAS-HAR/TCBL/NanoHAR line, tied to the removed synthetic-results era | ⬜ mark as legacy/superseded; BASS-HAR P1 outline to be created post-meeting |

## B. Missing highest-value artifact: the Paper-1 skeleton

No BASS-HAR paper outline exists anywhere. The pilot already contains a
publication-shaped result (evidence-mechanism ladder + supervision-gap figure +
two benchmark/methodology contributions). A P1 skeleton (title options, venue
shortlist, section stubs, results table from RESULTS.md, the label-fraction
curve as headline figure) is the top post-meeting artifact — **after** the
Phase B v1 vs semi-supervised framing decision, which determines the paper's
central claim.

## C. Reproducibility package gaps (pilot folder)

- No `requirements.txt` / environment pinning → ✅ added 2026-09-20
  (torch 2.6.0+cu124, scikit-learn 1.8.0, scipy 1.17.0, numpy 2.2.6,
  pandas 2.3.3, matplotlib 3.10.8, Python 3.13, CUDA 12.4)
- Constants hardcoded across runners (scales, thresholds, seeds) → config
  consolidation queued for P1 cycle
- No per-run artifact manifest (which log = which config) → P1 cycle

## D. Data lineage undocumented

Datasets on disk have no provenance record (source URL, version, download
date, preprocessing applied). → ✅ `05_Code/PhD-HAR-Segmentation/data/DATA_MANIFEST.md`
stub created with known facts + TODO fields to complete.

## E. Round-1 carried items — status

- §1 substantive proposal revisions (§1.7/§3.5/§3.6/§3.8) — ⬜ meeting decision
- §4.2 dataset re-scoping (PAMAP2 demotion / Opportunity primary) — ⬜ meeting decision
- §3.5 subsection renumbering (5.1–5.3) — ⬜ bundle with substantive edit
- Mid-tier 2019–22 reference recency audit — ⬜ scheduled next revision cycle
- P1 empirical ladder (LOSO / natural transitions / baselines / ablations) — ⬜ post-meeting program

## Round-2 sequencing

**Done today:** README refresh, execution-plan status block, pilot
requirements.txt, data manifest stub, this audit — all committed to the private
workspace repo.
**Meeting:** unchanged (Phase B v1 vs pivot; dataset re-scope; substantive
proposal edits sign-off).
**Post-meeting:** P1 skeleton → execution-plan rebase → P1 ladder (weeks 1–4).
