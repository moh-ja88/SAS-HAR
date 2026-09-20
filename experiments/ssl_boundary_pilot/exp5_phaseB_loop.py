"""Phase B loop v0: label-free boundary refinement via self-training.

Closes the supervision gap (RESULTS.md next-step #1):
  iter 0: unsupervised MMD evidence (2 s sides) -> pseudo-cuts (robust-z > 2.5, NMS 0.5 s)
  iter k: cut-supervised head trained on pseudo-cuts -> head evidence -> new pseudo-cuts

NO ground truth is used anywhere in the method (thresholds are fixed a priori).
Test GT is used only for reporting. Train-stream cut precision is a legitimate
self-diagnostic (we constructed those streams). Oracle-threshold F1 reported as
a reference line only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, r"C:\Work\OpenCode\PhDYasmeen\04_Experiments\Pilot_SSL_boundary")
import pamap2_stream as ps
import pilot_concat as pc
from evaluate_boundary import match_f1, peak_pick
from exp2_distributional import frame_embeddings, mmd_curve, robust_z, to_samples
from exp3_trained_head import (FEAT_DIM, SIDE, GAP, features_at, train_head)
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import average_precision_score
from train_ssl import train_ssl

MMD_SIDE, MMD_GAP = 2.0, 0.25      # best val scale from exp2
PSEUDO_Z = 2.5                       # label-free MMD peak threshold
HEAD_PCT = 98.5                      # label-free head peak percentile
N_ITER = 4
HERE = Path(__file__).parent


@torch.no_grad()
def head_evidence_batched(net, Z):
    """Head logits at every valid grid position, batched."""
    dev = next(net.parameters()).device
    G0, G1 = GAP + SIDE, len(Z) - GAP - SIDE
    F = np.stack([features_at(Z, g) for g in range(G0, G1)])
    chunks = [net(torch.as_tensor(F[i:i + 1024], dtype=torch.float32, device=dev))
              .squeeze(-1).cpu().numpy() for i in range(0, len(F), 1024)]
    out = np.zeros(len(Z), dtype=np.float32)
    out[G0:G0 + len(F)] = np.concatenate(chunks)
    return out


def head_curve_features(Z):
    return np.stack([f for f in (features_at(Z, g)
                                 for g in range(GAP + SIDE, len(Z) - GAP - SIDE))])


def evaluate_test(ev_frames, gt, T, z_thr):
    ev = robust_z(gaussian_filter1d(to_samples(ev_frames, T), 5))
    nm = np.zeros(T, dtype=bool)
    for g in gt:
        nm[max(0, g - 25):g + 26] = True
    ap = average_precision_score(nm.astype(int), ev)
    p, r, f1, mae, tp = match_f1(peak_pick(ev, z_thr), gt, 25)
    # oracle threshold (reference only)
    best = -1
    for pct in range(90, 100):
        t = np.percentile(ev, pct)
        _, _, fo, _, _ = match_f1(peak_pick(ev, t), gt, 25)
        best = max(best, fo)
    return dict(ap=float(ap), f1=float(f1), p=p, r=r, me_s=mae, f1_oracle=best,
                n_gt=len(gt))


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    rng = np.random.default_rng(seed)
    subjects, summs, stats = ps.load_all(pc.DATA)
    tr = [c for s in ps.TRAIN_SUBJ for c in subjects[s]]
    sm = [m for s in ps.TRAIN_SUBJ for m in summs[s]]
    tr_x, _ = ps.make_training_windows(tr, sm, stats, seed)
    va_x, _ = ps.make_training_windows(subjects[4], summs[4], stats, seed)
    model, _ = train_ssl(tr_x, va_x, epochs=30, seed=seed)
    for p in model.parameters():
        p.requires_grad_(False)

    # --- fixed streams: 2 shuffles per train subject + val + test ---
    streams = []
    for sid in ps.TRAIN_SUBJ:
        for _ in range(2):
            x, gt = pc.build_concat(subjects[sid], summs[sid], stats, rng)
            xn = (x - stats["mean"]) / stats["std"]
            Z = frame_embeddings(model, xn)
            streams.append(dict(sid=sid, T=len(x), gt=gt, Z=Z))
    vx, vgt = pc.build_concat(subjects[4], summs[4], stats, rng)
    vxn = (vx - stats["mean"]) / stats["std"]
    vZ = frame_embeddings(model, vxn)
    tx, tgt = pc.build_concat(subjects[7], summs[7], stats, rng)
    txn = (tx - stats["mean"]) / stats["std"]
    tZ = frame_embeddings(model, txn)
    n_true_total = sum(len(s["gt"]) for s in streams)
    print(f"{len(streams)} train streams, {n_true_total} true cuts (diag only); "
          f"test {len(tgt)} GT")

    # --- iteration 0: MMD pseudo-cuts (kept permanently as recall floor) ---
    mmd_cuts = {}
    pseudo = {}
    for i, s in enumerate(streams):
        mmd = mmd_curve(s["Z"], side_s=MMD_SIDE, gap_s=MMD_GAP)
        ev = robust_z(gaussian_filter1d(to_samples(mmd, s["T"]), 5))
        mmd_cuts[i] = peak_pick(ev, PSEUDO_Z)
        pseudo[i] = mmd_cuts[i]
    n0 = sum(len(v) for v in pseudo.values())
    diag = []
    for i, s in enumerate(streams):
        tp = match_f1(pseudo[i], s["gt"], 25)[4]     # tp = index 4
        diag.append(tp / max(1, len(pseudo[i])))
    mmd_test = evaluate_test(mmd_curve(tZ, side_s=MMD_SIDE, gap_s=MMD_GAP),
                             tgt, len(tx), PSEUDO_Z)
    print(f"\niter 0 (MMD): pseudo-cuts={n0} (true={n_true_total}, "
          f"prec={np.mean(diag):.2f}) | TEST AUPRC={mmd_test['ap']:.3f} "
          f"F1={mmd_test['f1']:.3f} oracle={mmd_test['f1_oracle']:.3f}")

    history = [dict(iter=0, n_pseudo=n0, pseudo_prec=float(np.mean(diag)),
                    test=mmd_test)]

    # --- iterations 1..N: retrain head on pseudo-cuts ---
    for it in range(1, N_ITER + 1):
        X, y = [], []
        for i, s in enumerate(streams):
            Tf = len(s["Z"])
            gtf = pseudo[i] // 4
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
                f = features_at(s["Z"], g)
                if f is not None:
                    X.append(f)
                    y.append(1 if g in pos else 0)
        X = np.asarray(X, np.float32)
        y = np.asarray(y, np.int64)
        # val stream for early stop (labels = proximity to MMD peaks, not GT)
        vpos = set()
        vev = robust_z(gaussian_filter1d(to_samples(
            mmd_curve(vZ, side_s=MMD_SIDE, gap_s=MMD_GAP), len(vx)), 5))
        for g0 in peak_pick(vev, PSEUDO_Z) // 4:
            for off in range(-5, 6, 2):
                vpos.add(int(g0 + off))
        Xv, yv = [], []
        for g in range(GAP + SIDE, len(vZ) - GAP - SIDE):
            if len(vpos) > 0:
                Xv.append(features_at(vZ, g))
                yv.append(1 if g in vpos else 0)
        net = train_head(X, y, np.asarray(Xv, np.float32),
                         np.asarray(yv, np.int64), seed + it, epochs=6)

        # head evidence -> next pseudo-cuts + test eval (batched, percentile thr)
        new_pseudo, diags = [], []
        for i, s in enumerate(streams):
            hc = head_evidence_batched(net, s["Z"])
            ev = robust_z(gaussian_filter1d(to_samples(hc, s["T"]), 5))
            cuts = peak_pick(ev, np.percentile(ev, HEAD_PCT))
            new_pseudo.append(cuts)
            tp = match_f1(cuts, s["gt"], 25)[4]
            diags.append(tp / max(1, len(cuts)))
        hc_t = head_evidence_batched(net, tZ)
        ev_t = robust_z(gaussian_filter1d(to_samples(hc_t, len(tx)), 5))
        te = evaluate_test(hc_t, tgt, len(tx),
                           z_thr=np.percentile(ev_t, HEAD_PCT))
        n_new = sum(len(v) for v in new_pseudo)
        print(f"iter {it} (head): pseudo-cuts={n_new} "
              f"({n_new/max(1,n0):.2f}x of iter0) prec={np.mean(diags):.2f} | "
              f"TEST AUPRC={te['ap']:.3f} F1={te['f1']:.3f} "
              f"oracle={te['f1_oracle']:.3f}", flush=True)
        history.append(dict(iter=it, n_pseudo=n_new,
                            pseudo_prec=float(np.mean(diags)), test=te))
        # candidate maintenance: MMD recall floor + head peaks (NMS-merged)
        pseudo = {}
        for i, s in enumerate(streams):
            merged = np.sort(np.concatenate([mmd_cuts[i], new_pseudo[i]]))
            keep = []
            for c in merged:
                if not keep or c - keep[-1] > 25:   # NMS 0.5 s
                    keep.append(c)
            pseudo[i] = np.asarray(keep, dtype=int)
        # drift guards
        n_m = sum(len(v) for v in pseudo.values())
        if n_m > 4 * n0 or n_m < 0.15 * n0:
            print(f"  drift guard triggered (pool={n_m}) — stopping", flush=True)
            break

    (HERE / "results" / f"phaseB_seed{seed}.json").write_text(
        json.dumps(history, indent=1, default=float))
    print("\nsaved results/phaseB_seed%d.json" % seed)


if __name__ == "__main__":
    main()
