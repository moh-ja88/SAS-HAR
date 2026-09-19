"""Concatenated-stream H1 test (DESIGN.md addendum v2).

Discovery (2026-09-18): PAMAP2 protocol labels cannot provide sample-accurate
boundary GT -- postural changes happen at unknown positions inside long NULL
pauses; label edges bracket them only loosely (derivative peaks within +-0.5s
of edges: 0-4%; wide-window correlation negative).

Pilot protocol (standard CPD benchmark methodology):
  1. per subject: take valid activity runs, trim TRIM_S ease-in/out at both ends
  2. cut into random-length segments SEG_S ~ U[a,b]
  3. shuffle segments, concatenate -> continuous stream with EXACT cut GT
  4. SSL trained only on train-subject pure windows (unchanged)
  5. evidence on test-subject concat streams; GT = cut positions
"""
from __future__ import annotations

import sys

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import average_precision_score

sys.path.insert(0, r"C:\Work\OpenCode\PhDYasmeen\04_Experiments\Pilot_SSL_boundary")
import pamap2_stream as ps
from evaluate_boundary import derivative_curve, peak_pick, match_f1
from ssl_model import evidence_curves
from train_ssl import train_ssl

DATA = r"C:\Work\OpenCode\PhDYasmeen\05_Code\PhD-HAR-Segmentation\data\pamap2\PAMAP2_Dataset\Protocol"
TRIM_S = 2.5          # drop ease-in/out at activity edges
SEG_MIN, SEG_MAX = 4, 30   # segment length bounds [s]
NEAR, TOL = 25, 25    # 0.5 s


def robust_z(e):
    med, mad = np.median(e), np.median(np.abs(e - np.median(e))) + 1e-9
    return (e - med) / (1.4826 * mad)


def build_concat(chunks, summs, stats, rng):
    """Valid activity runs -> pieces -> SHUFFLED concat stream with exact GT.

    Correction 2026-09-19: the original implementation appended pieces in
    chronological order, so most cuts were WITHIN-activity (no distribution
    change) -- invalidating all evidence families tested against them (the
    probe-contrast falsification test exposed this: it detected only the
    genuine activity changes, 11/141). Now pieces are shuffled; adjacent
    same-label pieces are re-ordered away greedily; GT = cuts between
    differing labels only.
    """
    pieces, labels = [], []
    for c, m in zip(chunks, summs):
        for st, en, lab in m["runs"]:
            if lab == ps.NULL_LABEL or (en - st) < (TRIM_S * 2 + SEG_MIN) * ps.FS:
                continue
            st2, en2 = st + int(TRIM_S * ps.FS), en - int(TRIM_S * ps.FS)
            seg = c["x"][st2:en2]
            i = 0
            while i < len(seg):
                L = int(rng.uniform(SEG_MIN, SEG_MAX) * ps.FS)
                piece = seg[i:i + L]
                if len(piece) >= SEG_MIN * ps.FS:
                    pieces.append(piece)
                    labels.append(lab)
                i += L
    order = rng.permutation(len(pieces))
    # greedy pass: avoid same-label adjacency where possible
    final = []
    for idx in order:
        if final and labels[final[-1]] == labels[idx]:
            for j in range(len(final) - 2, -1, -1):
                if labels[final[j]] != labels[idx] and \
                   (j + 1 == len(final) or labels[final[j + 1]] != labels[idx]):
                    final.insert(j + 1, idx)
                    break
            else:
                final.append(idx)
        else:
            final.append(idx)
    stream, gt, t = [], [], 0
    prev_lab = None
    for idx in final:
        stream.append(pieces[idx])
        t += len(pieces[idx])
        if prev_lab is not None and labels[idx] != prev_lab:
            gt.append(t)
        prev_lab = labels[idx]
    x = np.concatenate(stream, axis=0)
    return x, np.asarray(gt, dtype=int)


def near_mask(gt, n):
    m = np.zeros(n, dtype=bool)
    for g in gt:
        m[max(0, g - NEAR):g + NEAR + 1] = True
    return m


def eval_stream(name, eps, gt, thr=None):
    eps = robust_z(gaussian_filter1d(eps, sigma=5))
    nm = near_mask(gt, len(eps))
    r = np.corrcoef(nm.astype(float), eps)[0, 1]
    ap = average_precision_score(nm.astype(int), eps)
    line = f"{name:10s} r={r:+.3f} AUPRC={ap:.3f} (chance={nm.mean():.3f})"
    if thr is not None:
        dets = peak_pick(eps, thr)
        p, rc, f1, mae, tp = match_f1(dets, gt, TOL)
        line += f" | thr-F1={f1:.3f} P={p:.2f} R={rc:.2f} MAE={mae:.2f}s"
    print(line)
    return r, ap


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    rng = np.random.default_rng(seed)
    subjects, summs, stats = ps.load_all(DATA)

    tr = [c for s in ps.TRAIN_SUBJ for c in subjects[s]]
    sm = [m for s in ps.TRAIN_SUBJ for m in summs[s]]
    tr_x, _ = ps.make_training_windows(tr, sm, stats, seed)
    va_x, _ = ps.make_training_windows(subjects[4], summs[4], stats, seed)
    model, vloss = train_ssl(tr_x, va_x, epochs=30, seed=seed)
    print(f"seed {seed}, val loss {vloss:.4f}")

    # threshold picked on VAL subject concat stream (CP head)
    vx, vgt = build_concat(subjects[4], summs[4], stats, rng)
    ecp_v, emt_v = evidence_curves(model, (vx - stats["mean"]) / stats["std"])
    best = (None, -1)
    for pct in range(90, 100):
        t = np.percentile(robust_z(gaussian_filter1d(ecp_v, 5)), pct)
        dets = peak_pick(robust_z(gaussian_filter1d(ecp_v, 5)), t)
        _, _, f1, _, _ = match_f1(dets, vgt, TOL)
        if f1 > best[1]:
            best = (t, f1)
    print(f"val concat: {len(vgt)} GT, thr={best[0]:.2f}z, val F1={best[1]:.3f}\n")

    for sid in ps.TEST_SUBJ:
        x, gt = build_concat(subjects[sid], summs[sid], stats, rng)
        if len(gt) == 0:
            print(f"S{sid}: no usable GT"); continue
        xn = (x - stats["mean"]) / stats["std"]
        ecp, emt = evidence_curves(model, xn)
        print(f"--- S{sid}: {len(x)/50:.0f}s stream, {len(gt)} boundaries ---")
        eval_stream("CP", ecp, gt, best[0])
        eval_stream("MTM", emt, gt, best[0])
        eval_stream("CP+MTM", robust_z(ecp) + robust_z(emt), gt, best[0])
        eval_stream("B0 deriv", derivative_curve(xn), gt, best[0])


if __name__ == "__main__":
    main()
