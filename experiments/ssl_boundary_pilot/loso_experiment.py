"""LOSO + label-fraction experiment — implements LOSO_DESIGN.md (frozen 2026-09-20).

8-fold leave-one-subject-out over PAMAP2 subjects 1-8. Per fold:
  SSL retrained on 7 train subjects; held-out subject becomes a concat stream
  (exact GT). Families: B0 derivative, CP, MTM, MMD(2s, label-free 2.5z),
  trained head (label-free p98.5), probe-contrast at k in {1,2,4,7} labeled
  subjects (AUPRC + oracle-F1 + classification accuracy).
Metrics: AUPRC (primary), oracle-F1 (upper bound), fixed-threshold F1 where a
label-free rule exists. Statistics: mean+-std + paired Wilcoxon vs B0.
Crash-safe: per-fold JSON in results/loso/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats as sps
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score

sys.path.insert(0, r"C:\Work\OpenCode\PhDYasmeen\04_Experiments\Pilot_SSL_boundary")
import pamap2_stream as ps
import pilot_concat as pc
from evaluate_boundary import derivative_curve, match_f1, peak_pick
from exp2_distributional import frame_embeddings, mmd_curve, robust_z, to_samples
from exp3_trained_head import FEAT_DIM, GAP, SIDE, features_at, train_head
from exp4_probe_contrast import gap_embed, js, windows_left_right
from exp5_phaseB_loop import head_evidence_batched
from ssl_model import evidence_curves
from train_ssl import train_ssl

ALL_SUBJ = [1, 2, 3, 4, 5, 6, 7, 8]
TOL = 25
HERE = Path(__file__).parent
OUT = HERE / "results" / "loso"
K_SETS = [1, 2, 4, 7]


def pure_windows(subjects, summs, stats, sids):
    xs, ys = [], []
    for sid in sids:
        for c, m in zip(subjects[sid], summs[sid]):
            xc = (c["x"] - stats["mean"]) / stats["std"]
            for st, en, lab in m["runs"]:
                if lab == 0 or en - st < 200:
                    continue
                for w0 in range(st, en - 200 + 1, 50):
                    xs.append(xc[w0:w0 + 200])
                    ys.append(lab)
    return np.asarray(xs, np.float32).transpose(0, 2, 1), np.asarray(ys)


def near_mask(gt, n):
    m = np.zeros(n, dtype=bool)
    for g in gt:
        m[max(0, g - TOL):g + TOL + 1] = True
    return m


def oracle_f1(ev, gt):
    best = -1.0
    for pct in np.arange(90.0, 100.0, 0.5):
        _, _, f1, _, _ = match_f1(peak_pick(ev, np.percentile(ev, pct)), gt, TOL)
        best = max(best, f1)
    return best


def metrics(ev_raw, gt, T, fixed_thr=None):
    ev = robust_z(gaussian_filter1d(ev_raw, 5))
    nm = near_mask(gt, T)
    ap = float(average_precision_score(nm.astype(int), ev))
    of1 = float(oracle_f1(ev, gt))
    out = dict(auprc=ap, f1_oracle=of1, chance=float(nm.mean()))
    if fixed_thr is not None:
        p, r, f1, mae, _ = match_f1(peak_pick(ev, fixed_thr), gt, TOL)
        out.update(f1_fixed=f1, p=p, r=r, me_s=None if mae != mae else mae)
    return out


def head_training_set(model, subjects, summs, stats, train_sids, rng):
    X, y = [], []
    for sid in train_sids:
        x, gt = pc.build_concat(subjects[sid], summs[sid], stats, rng)
        xn = (x - stats["mean"]) / stats["std"]
        Z = frame_embeddings(model, xn)
        ev = robust_z(gaussian_filter1d(to_samples(
            mmd_curve(Z, side_s=2.0, gap_s=0.25), len(x)), 5))
        cuts = peak_pick(ev, 2.5)
        gtf = cuts // 4
        Tf = len(Z)
        pos = set()
        for g0 in gtf:
            for off in range(-5, 6, 2):
                pos.add(int(g0 + off))
        pos = [g for g in pos if GAP + SIDE <= g < Tf - GAP - SIDE]
        far = np.ones(Tf, dtype=bool)
        for g0 in gtf:
            far[max(0, g0 - 19):g0 + 19] = False
        cand = np.where(far)[0]
        cand = cand[(cand > GAP + SIDE) & (cand < Tf - GAP - SIDE)]
        if len(cand) < len(pos):
            continue
        neg = rng.choice(cand, size=min(2 * len(pos), len(cand)), replace=False)
        for g in list(pos) + list(neg):
            f = features_at(Z, g)
            if f is not None:
                X.append(f)
                y.append(1 if g in pos else 0)
    return np.asarray(X, np.float32), np.asarray(y, np.int64)


def run_fold(fold, test_sid, subjects, summs, stats):
    seed = 42 + fold
    rng = np.random.default_rng(seed)
    train_sids = [s for s in ALL_SUBJ if s != test_sid]

    tr_x, _ = ps.make_training_windows(
        [c for s in train_sids for c in subjects[s]],
        [m for s in train_sids for m in summs[s]], stats, seed)
    model, vloss = train_ssl(tr_x, tr_x[:1024], epochs=30, seed=seed)

    x, gt = pc.build_concat(subjects[test_sid], summs[test_sid], stats, rng)
    xn = (x - stats["mean"]) / stats["std"]
    T = len(x)
    res = dict(fold=fold, test_subject=test_sid, val_loss=float(vloss),
               n_gt=int(len(gt)), T=int(T), families={})

    # B0, CP, MTM
    res["families"]["B0"] = metrics(derivative_curve(xn), gt, T)
    ecp, emt = evidence_curves(model, xn)
    res["families"]["CP"] = metrics(ecp, gt, T)
    res["families"]["MTM"] = metrics(emt, gt, T)

    # MMD (label-free 2.5z)
    Z = frame_embeddings(model, xn)
    mmd = to_samples(mmd_curve(Z, side_s=2.0, gap_s=0.25), T)
    res["families"]["MMD2s"] = metrics(mmd, gt, T, fixed_thr=2.5)

    # trained head (label-free p98.5)
    Xh, yh = head_training_set(model, subjects, summs, stats, train_sids, rng)
    net = train_head(Xh, yh, Xh[:2048], yh[:2048], seed, epochs=6)
    hc = head_evidence_batched(net, Z)
    hc_s = to_samples(hc, T)
    ev_h = robust_z(gaussian_filter1d(hc_s, 5))
    res["families"]["head"] = metrics(hc_s, gt, T,
                                      fixed_thr=float(np.percentile(ev_h, 98.5)))

    # probe-contrast label fractions
    emb = {}
    labs = {}
    for sid in train_sids:
        W, y = pure_windows(subjects, summs, stats, [sid])
        emb[sid] = gap_embed(model, W)
        labs[sid] = y
    ts, L, R = windows_left_right(xn)
    eL = gap_embed(model, L)
    eR = gap_embed(model, R)
    Wt, yt = pure_windows(subjects, summs, stats, [test_sid])
    eT = gap_embed(model, Wt)

    def probe_evidence(probe):
        pL = probe.predict_proba(eL)
        pR = probe.predict_proba(eR)
        ev = np.zeros(T)
        for t, v in zip(ts, js(pL, pR)):
            ev[max(0, t - 6):t + 7] = v
        return ev

    res["label_fraction"] = {}
    for k in K_SETS:
        ks = rng.choice(train_sids, size=k, replace=False)
        Xp = np.concatenate([emb[s] for s in ks])
        yp = np.concatenate([labs[s] for s in ks])
        probe = LogisticRegression(max_iter=2000).fit(Xp, yp)
        ev = probe_evidence(probe)
        m = metrics(ev, gt, T)
        m["cls_acc"] = float(accuracy_score(yt, probe.predict(eT)))
        m["k_subjects"] = [int(s) for s in ks]
        res["label_fraction"][f"k{k}"] = m

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"fold_s{test_sid}.json").write_text(json.dumps(res, indent=1,
                                                           default=float))
    line = f"S{test_sid}: GT={len(gt)} " + " ".join(
        f"{k}={v['auprc']:.3f}" for k, v in res["families"].items())
    lk = " ".join(f"k{k}={res['label_fraction'][f'k{k}']['auprc']:.3f}"
                  for k in K_SETS)
    print(f"fold {fold} {line} | {lk}", flush=True)
    return res


def aggregate(results):
    fams = ["B0", "CP", "MTM", "MMD2s", "head"]
    agg = {}
    for f in fams:
        aps = [r["families"][f]["auprc"] for r in results]
        ofs = [r["families"][f]["f1_oracle"] for r in results]
        agg[f] = dict(auprc_mean=float(np.mean(aps)), auprc_std=float(np.std(aps)),
                      f1_oracle_mean=float(np.mean(ofs)), per_subject=aps)
    chance = float(np.mean([r["families"]["B0"]["chance"] for r in results]))
    agg["chance_mean"] = chance
    # label fractions
    agg["label_fraction"] = {}
    for k in K_SETS:
        aps = [r["label_fraction"][f"k{k}"]["auprc"] for r in results]
        ofs = [r["label_fraction"][f"k{k}"]["f1_oracle"] for r in results]
        acc = [r["label_fraction"][f"k{k}"]["cls_acc"] for r in results]
        agg["label_fraction"][f"k{k}"] = dict(
            auprc_mean=float(np.mean(aps)), auprc_std=float(np.std(aps)),
            f1_oracle_mean=float(np.mean(ofs)), cls_acc_mean=float(np.mean(acc)))
    # Wilcoxon vs B0
    agg["wilcoxon_vs_B0"] = {}
    for f in fams[1:]:
        try:
            w = sps.wilcoxon(agg[f]["per_subject"], agg["B0"]["per_subject"])
            agg["wilcoxon_vs_B0"][f] = dict(p=float(w.pvalue))
        except ValueError:
            agg["wilcoxon_vs_B0"][f] = dict(p=float("nan"))
    # criteria
    lf = agg["label_fraction"]
    agg["criteria"] = dict(
        S1_probe7_robust=bool(sum(1 for r in results
                                  if r["label_fraction"]["k7"]["auprc"] >
                                  r["families"]["B0"]["chance"]) >= 7
                              and lf["k7"]["auprc_mean"] >= 2 * chance),
        S2_k1_beats_MMD=bool(lf["k1"]["auprc_mean"] >= agg["MMD2s"]["auprc_mean"]),
        S3_ordering=bool(lf["k7"]["auprc_mean"] > agg["head"]["auprc_mean"] >
                         agg["MMD2s"]["auprc_mean"] > agg["CP"]["auprc_mean"] >
                         max(agg["MTM"]["auprc_mean"], agg["B0"]["auprc_mean"])))
    return agg


def main():
    only = [int(a) for a in sys.argv[1:]] or None
    subjects, summs, stats = ps.load_all(pc.DATA)
    results = []
    for fold, sid in enumerate(ALL_SUBJ):
        if only and sid not in only:
            continue
        results.append(run_fold(fold, sid, subjects, summs, stats))
    if not results:
        return
    agg = aggregate(results)
    (OUT / "aggregate.json").write_text(json.dumps(agg, indent=1, default=float))
    print("\n=== aggregate (mean±std AUPRC; chance %.3f) ===" % agg["chance_mean"])
    for f in ["B0", "CP", "MTM", "MMD2s", "head"]:
        a = agg[f]
        p = agg["wilcoxon_vs_B0"].get(f, {}).get("p", float("nan"))
        print(f"{f:6s} AUPRC {a['auprc_mean']:.3f}±{a['auprc_std']:.3f} "
              f"oracleF1 {a['f1_oracle_mean']:.3f}  p(B0)={p:.4f}")
    for k in K_SETS:
        a = agg["label_fraction"][f"k{k}"]
        print(f"probe k={k}: AUPRC {a['auprc_mean']:.3f}±{a['auprc_std']:.3f} "
              f"oracleF1 {a['f1_oracle_mean']:.3f} clsAcc {a['cls_acc_mean']:.3f}")
    print("criteria:", agg["criteria"])


if __name__ == "__main__":
    main()
