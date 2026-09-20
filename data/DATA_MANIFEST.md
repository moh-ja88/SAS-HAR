# Dataset Manifest — provenance & preprocessing record
**Rule:** every dataset used in a thesis experiment gets a row here BEFORE first use.
Last updated: 2026-09-20.

| Dataset | Location (this repo) | Form on disk | Source | Version / files | Downloaded | Preprocessing applied | Used by |
|---|---|---|---|---|---|---|---|
| PAMAP2 | `pamap2/PAMAP2_Dataset/Protocol/subject101–109.dat` | raw .dat, 100 Hz, 54 cols | official PAMAP2 (UCI ML repo mirror) | Protocol only (9 subjects) | ⬜ record date | pilot: 18 IMU ch (acc+gyro ×3 placements), resample→50 Hz, NaN-row drop, gap split >0.1 s, z-score (train stats) — see `04_Experiments/Pilot_SSL_boundary/pamap2_stream.py` | SSL boundary pilot; subject 109 excluded (95% NaN) |
| Opportunity | `opportunity/` (raw .dat) + processed npz (e.g. `opportunity_body_worn_*.npz`) | raw ADL/drill runs + windowed npz | official Opportunity challenge | ⬜ enumerate | ⬜ record | legacy npz pipeline (unverified); pilot-grade loader NOT yet built | planned: natural-transition validation |
| UCI-HAR | `../Signals/` + `../Dataset/` (repo root dirs) | pre-windowed inertial signals (561-feat + raw windows) | UCI ML repo | 30 subjects, 6 activities, fixed windows | ⬜ record date | none by us; **no transitions — windowed at source** | reference only (proposal §4.2 excludes as primary) |
| WISDM | `wisdm/` | v2.0 raw (phone+watch, accel+gyro, user ids) | official WISDM lab | v2.0 actitracker raw (NOT v1.1 — notebook mismatch noted) | ⬜ record date | none yet | WISDM notebooks (Colab-oriented) |
| UCI-HAPT | ⬜ not on disk | — | UCI | explicit postural-transition labels | — | — | planned natural-transition source (needs download) |

**Open items:** (1) fill download dates + exact file inventories; (2) verify Opportunity
npz provenance before P1 use (they predate this workspace's version control); (3) decide
UCI-HAPT download for natural-transition experiments; (4) document the concat-benchmark
construction (PAMAP2 → shuffled exact-GT streams) in the P1 paper's reproducibility section
— code: `04_Experiments/Pilot_SSL_boundary/pilot_concat.py`.
