"""Opportunity natural-transition validation — implements OPPORTUNITY_DESIGN.md.

4-fold LOSO over subjects S1-S4 (ADL runs only). Streams: 45 body-IMU channels
@30 Hz, NaN-dropped, gap-split. GT: locomotion (col 243) label changes between
valid labels. Families: B0/CP/MTM/MMD(2s)/probe-contrast k in {1,2,3}.
Tolerances: primary 2 s (annotation lag, justified in design), secondary 0.5 s.
Crash-safe per-fold JSON in results/opportunity/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats as sps
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score

sys.path.insert(0, r"C:\Work\OpenCode\PhDYasmeen\04_Experiments\Pilot_SSL_boundary")
from evaluate_boundary import derivative_curve, match_f1, peak_pick
from exp2_distributional import frame_embeddings, mmd2_rbf, robust_z
from ssl_model import evidence_curves
from train_ssl import train_ssl

ROOT = Path(r"C:\Work\OpenCode\PhDYasmeen\05_Code\PhD-HAR-Segmentation\data"
            r"\opportunity\raw\OpportunityUCIDataset\dataset")
FS = 30
N_CH = 45
WIN = 120            # 4 s
HOP = 30             # 1 s
TOL_S, TOL_P = 60, 15   # 2 s primary, 0.5 s secondary
SUBJECTS = [1, 2, 3, 4]
HERE = Path(__file__).parent
OUT = HERE / "results" / "opportunity"


def imu_cols():
    cols = []
    for start in (37, 50, 63, 76, 89):
        cols += list(range(start, start + 9))
    return cols


COLS = imu_cols()
LOCO = 243


def load_run(subj, run):
    """One stream per ADL run. No gap-splitting (MILLISEC jumps are sensor
    artifacts; splitting shatters label runs). Seam positions (NaN-drop
    boundaries >3 rows) are masked out downstream instead."""
    p = ROOT / f"S{subj}-ADL{run}.dat"
    df = pd.read_csv(p, sep=r"\s+", header=None, engine="c", skiprows=1)
    arr = df.to_numpy(float)
    X, lab = arr[:, COLS], arr[:, LOCO]
    ok = ~np.isnan(X).any(1)
    X, lab = X[ok], np.nan_to_num(lab[ok]).astype(int)
    bad = ~ok
    seams = np.where(np.convolve(bad.astype(int), np.ones(4), mode="valid") >= 3)[0]
    seam_mask = np.zeros(len(X) + len(bad), dtype=bool)
    seam_mask[seams] = True
    # map seam positions (original index space) back through ok-filter:
    # recompute directly on filtered arrays is complex; approximate by marking
    # neighborhoods in original space then filtering
    full = np.zeros(len(arr), dtype=bool)
    for s in seams:
        full[max(0, s - 2):s + 4] = True
    seam = full[ok]
    return [dict(x=X, label=lab, seam=seam)]


def load_subject(subj):
    out = []
    for run in range(1, 6):
        out += load_run(subj, run)
    return out


def runs_of(label):
    ch = np.where(np.diff(label) != 0)[0] + 1
    b = np.concatenate([[0], ch, [len(label)]])
    return [(int(b[i]), int(b[i + 1]), int(label[b[i]])) for i in range(len(b) - 1)]


def edges_of(runs):
    """label-change edges between valid (non-null) labels."""
    out = []
    for (s1, e1, l1), (s2, e2, l2) in zip(runs[:-1], runs[1:]):
        if l1 != 0 and l2 != 0 and s2 == e1:
            out.append(e1)
    return np.asarray(out, dtype=int)


def valid_mask(runs, T):
    m = np.zeros(T, dtype=bool)
    for s, e, l in runs:
        if l != 0:
            m[s:e] = True
    return m


def make_windows(chunks, stats, seed, valid_only=True):
    rng = np.random.default_rng(seed)
    xs = []
    for c in chunks:
        xc = (c["x"] - stats["mean"]) / stats["std"]
        runs = runs_of(c["label"])
        for st, en, lab in runs:
            if valid_only and lab == 0:
                continue
            if en - st < WIN:
                continue
            for w0 in range(st, en - WIN + 1, HOP):
                xs.append(xc[w0:w0 + WIN])
    xs = np.asarray(xs, np.float32).transpose(0, 2, 1)
    return xs[rng.permutation(len(xs))]


def near_mask(gt, n, tol):
    m = np.zeros(n, dtype=bool)
    for g in gt:
        m[max(0, g - tol):g + tol + 1] = True
    return m


def oracle_f1(ev, gt, tol):
    best = -1.0
    for pct in np.arange(90.0, 100.0, 0.5):
        _, _, f1, _, _ = match_f1(peak_pick(ev, np.percentile(ev, pct), nms=tol),
                                  gt, tol)
        best = max(best, f1)
    return best


def metrics(ev_raw, gt, valid, tol):
    ev = robust_z(gaussian_filter1d(ev_raw, int(0.1 * FS)))
    nm = near_mask(gt, len(ev), tol) & valid
    ap = float(average_precision_score(nm.astype(int), ev * valid))
    of1 = float(oracle_f1(ev * valid, gt, tol))
    return dict(auprc=ap, f1_oracle=of1, chance=float(nm.sum() / max(1, valid.sum())))


def mmd_evidence(Z, T, side_s=2.0, gap_s=0.25, stride_s=0.5):
    """Z: [Tf, D] frame embeddings at 7.5 Hz (stride 4 @ 30 Hz)."""
    fpf = FS / 4
    side = int(side_s * fpf)
    gap = max(1, int(gap_s * fpf))
    stride = max(1, int(stride_s * fpf))
    Tf = len(Z)
    ev = np.zeros(Tf, dtype=np.float32)
    lo = gap + side
    for g in range(lo, Tf - lo, stride):
        L = Z[g - gap - side: g - gap]
        R = Z[g + gap: g + gap + side]
        v = mmd2_rbf(L, R)
        ev[g - stride // 2: g + stride // 2 + 1] = v
    out = np.repeat(ev, 4)
    if len(out) < T:
        out = np.pad(out, (0, T - len(out)), mode="edge")
    return out[:T]


@torch.no_grad()
def probe_evidence(model, probe, xn, T):
    dev = next(model.parameters()).device
    gap = 8   # ~0.25 s @30 Hz
    ts = np.arange(WIN + gap, T - WIN - gap, HOP)
    pL_all, pR_all = [], []
    Ls = np.stack([xn[t - gap - WIN: t - gap] for t in ts]).transpose(0, 2, 1)
    Rs = np.stack([xn[t + gap: t + gap + WIN] for t in ts]).transpose(0, 2, 1)
    for W in (Ls, Rs):
        embs = []
        for i in range(0, len(W), 256):
            z = model.encoder(torch.as_tensor(W[i:i + 256], dtype=torch.float32,
                                              device=dev))
            embs.append(z.mean(dim=2).cpu().numpy())
        p = probe.predict_proba(np.concatenate(embs))
        (pL_all if W is Ls else pR_all).append(p)
    pL, pR = pL_all[0], pR_all[0]
    m = 0.5 * (pL + pR)
    js = 0.5 * (pL * np.log(pL / m + 1e-9)).sum(1) + \
         0.5 * (pR * np.log(pR / m + 1e-9)).sum(1)
    ev = np.zeros(T)
    for t, v in zip(ts, js):
        ev[max(0, t - HOP // 2):t + HOP // 2 + 1] = v
    return ev


def pure_windows(subjects_data, stats, sids):
    xs, ys = [], []
    for sid in sids:
        for c in subjects_data[sid]:
            xc = (c["x"] - stats["mean"]) / stats["std"]
            for st, en, lab in runs_of(c["label"]):
                if lab == 0 or en - st < WIN:
                    continue
                for w0 in range(st, en - WIN + 1, HOP):
                    xs.append(xc[w0:w0 + WIN])
                    ys.append(lab)
    return np.asarray(xs, np.float32).transpose(0, 2, 1), np.asarray(ys)


def run_fold(fold, test_sid, data):
    seed = 42 + fold
    rng = np.random.default_rng(seed)
    train_sids = [s for s in SUBJECTS if s != test_sid]

    pool = np.concatenate([c["x"] for s in train_sids for c in data[s]])
    stats = dict(mean=pool.mean(0), std=pool.std(0) + 1e-8)

    tr_x = make_windows([c for s in train_sids for c in data[s]], stats, seed)
    model, vloss = train_ssl(tr_x, tr_x[:1024], epochs=30, seed=seed,
                             n_ch=N_CH, win=WIN)

    res = dict(fold=fold, test_subject=test_sid, val_loss=float(vloss),
               runs=[], label_fraction={})
    per_fam = {"B0": [], "CP": [], "MTM": [], "MMD2s": []}
    lf_aps = {1: [], 2: [], 3: []}

    # probe training embeddings per train subject (for k subsets)
    emb, labs = {}, {}
    for sid in train_sids:
        W, y = pure_windows(data, stats, [sid])
        embs = []
        with torch.no_grad():
            dev = next(model.parameters()).device
            for i in range(0, len(W), 256):
                z = model.encoder(torch.as_tensor(W[i:i + 256], dtype=torch.float32,
                                                  device=dev))
                embs.append(z.mean(dim=2).cpu().numpy())
        emb[sid] = np.concatenate(embs)
        labs[sid] = y

    for run_i, c in enumerate(data[test_sid]):
        T = len(c["x"])
        runs = runs_of(c["label"])
        gt = edges_of(runs)
        valid = valid_mask(runs, T) & ~c["seam"]
        if len(gt) == 0:
            continue
        xn = (c["x"] - stats["mean"]) / stats["std"]
        row = dict(run=run_i, T=int(T), n_gt=int(len(gt)))
        ecp, emt = evidence_curves(model, xn, win=WIN, hop=HOP)
        fams_ev = dict(B0=derivative_curve(xn), CP=ecp, MTM=emt)
        Z = frame_embeddings(model, xn, win=WIN, hop=HOP)
        fams_ev["MMD2s"] = mmd_evidence(Z, T)
        for fam, ev in fams_ev.items():
            for tol_name, tol in (("2s", TOL_S), ("0.5s", TOL_P)):
                m = metrics(ev, gt, valid, tol)
                row[f"{fam}_{tol_name}"] = m
            per_fam[fam].append(row[f"{fam}_2s"]["auprc"])
        # probe k-subsets
        for k in (1, 2, 3):
            ks = rng.choice(train_sids, size=k, replace=False)
            probe = LogisticRegression(max_iter=1500).fit(
                np.concatenate([emb[s] for s in ks]),
                np.concatenate([labs[s] for s in ks]))
            ev = probe_evidence(model, probe, xn, T)
            for tol_name, tol in (("2s", TOL_S), ("0.5s", TOL_P)):
                row[f"probe_k{k}_{tol_name}"] = metrics(ev, gt, valid, tol)
            lf_aps[k].append(row[f"probe_k{k}_2s"]["auprc"])
        res["runs"].append(row)
        print(f"  S{test_sid} run{run_i}: {len(gt)} GT | " + " ".join(
            f"{f}={row[f + '_2s']['auprc']:.3f}" for f in per_fam) +
            f" | probe_k3={row['probe_k3_2s']['auprc']:.3f}", flush=True)

    for fam, aps in per_fam.items():
        res[f"{fam}_auprc_mean"] = float(np.mean(aps)) if aps else None
    for k, aps in lf_aps.items():
        res[f"probe_k{k}_auprc_mean"] = float(np.mean(aps)) if aps else None
    res["chance_mean"] = float(np.mean(
        [r["B0_2s"]["chance"] for r in res["runs"]])) if res["runs"] else None

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"fold_s{test_sid}.json").write_text(json.dumps(res, indent=1,
                                                           default=float))
    line = f"fold {fold} S{test_sid}: " + " ".join(
        f"{f}={res[f'{f}_auprc_mean']:.3f}" for f in per_fam if res[f"{f}_auprc_mean"])
    lk = " ".join(f"k{k}={res[f'probe_k{k}_auprc_mean']:.3f}" for k in (1, 2, 3))
    print(line + f" | {lk} (chance {res['chance_mean']:.3f})", flush=True)
    return res


def main():
    only = [int(a) for a in sys.argv[1:]] or None
    data = {s: load_subject(s) for s in SUBJECTS}
    for s in SUBJECTS:
        mins = sum(len(c["x"]) for c in data[s]) / FS / 60
        edges = sum(len(edges_of(runs_of(c["label"]))) for c in data[s])
        print(f"S{s}: {len(data[s])} chunks, {mins:.0f} min, {edges} edges", flush=True)
    results = []
    for fold, sid in enumerate(SUBJECTS):
        if only and sid not in only:
            continue
        results.append(run_fold(fold, sid, data))
    if not results:
        return
    print("\n=== aggregate (2s tolerance, mean of per-subject means) ===")
    for f in ("B0", "CP", "MTM", "MMD2s"):
        vals = [r[f"{f}_auprc_mean"] for r in results
                if r[f"{f}_auprc_mean"] is not None]
        print(f"{f:6s} AUPRC {np.mean(vals):.3f}±{np.std(vals):.3f}")
    for k in (1, 2, 3):
        vals = [r[f"probe_k{k}_auprc_mean"] for r in results if r[f"probe_k{k}_auprc_mean"] is not None]
        print(f"probe k={k}: AUPRC {np.mean(vals):.3f}±{np.std(vals):.3f}")
    ch = np.mean([r["chance_mean"] for r in results])
    print(f"chance {ch:.3f}")


if __name__ == "__main__":
    main()
