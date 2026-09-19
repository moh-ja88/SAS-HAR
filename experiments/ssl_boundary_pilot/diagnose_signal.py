"""Signal diagnosis: does SSL evidence carry ANY boundary information?

Threshold-free analysis (immune to the sparse-GT threshold problem):
  1. robust per-subject z-norm of evidence (median/MAD)
  2. Gaussian smoothing sigma=0.1 s
  3. point-biserial r (near vs far), AUPRC (average precision with near-GT as positives)
  4. per-GT peak coverage: fraction of GT with a local evidence max within tolerance
     exceeding the p-th percentile of that subject's evidence
  5. same analysis for the derivative baseline B0 for reference
"""
from __future__ import annotations

import sys

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import average_precision_score

sys.path.insert(0, r"C:\Work\OpenCode\PhDYasmeen\04_Experiments\Pilot_SSL_boundary")
import pamap2_stream as ps
from evaluate_boundary import derivative_curve
from ssl_model import evidence_curves
from train_ssl import train_ssl

DATA = r"C:\Work\OpenCode\PhDYasmeen\05_Code\PhD-HAR-Segmentation\data\pamap2\PAMAP2_Dataset\Protocol"
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42
NEAR, TOL = 25, 25


def robust_z(e):
    med, mad = np.median(e), np.median(np.abs(e - np.median(e))) + 1e-9
    return (e - med) / (1.4826 * mad)


def near_mask(gt, n):
    m = np.zeros(n, dtype=bool)
    for g in gt:
        m[max(0, g - NEAR):g + NEAR + 1] = True
    return m


def diagnose(name, eps, gt, valid):
    eps = robust_z(gaussian_filter1d(eps, sigma=5))   # sigma 5 samples = 0.1 s
    nm = near_mask(gt, len(eps)) & valid
    far = valid & ~nm
    y = (nm & valid).astype(int)
    r = np.corrcoef(y[valid], eps[valid])[0, 1]
    ap = average_precision_score(y[valid], eps[valid])
    # per-GT peak coverage at several percentiles
    cov = {}
    for pct in (99.0, 99.5, 99.9):
        thr = np.percentile(eps[valid], pct)
        hit = 0
        for g in gt:
            lo, hi = max(0, g - TOL), min(len(eps), g + TOL + 1)
            if (eps[lo:hi] > thr).any():
                hit += 1
        cov[f"cov@p{pct}"] = hit / max(1, len(gt))
    dens = {f"p{p}": float(np.percentile(eps[valid], p)) for p in (50, 99, 99.9)}
    print(f"{name:12s} r={r:+.3f} AUPRC={ap:.4f} (chance={nm.sum()/valid.sum():.4f}) "
          f"cov={cov} med-to-p99.9 lift={dens['p99.9']-dens['p50']:.1f}z "
          f"near_med={np.median(eps[nm]):+.2f} far_med={np.median(eps[far]):+.2f}")
    return r, ap, cov


def main():
    subjects, summs, stats = ps.load_all(DATA)
    tr = [c for s in ps.TRAIN_SUBJ for c in subjects[s]]
    sm = [m for s in ps.TRAIN_SUBJ for m in summs[s]]
    tr_x, _ = ps.make_training_windows(tr, sm, stats, SEED)
    va_x, _ = ps.make_training_windows(subjects[4], summs[4], stats, SEED)
    model, vloss = train_ssl(tr_x, va_x, epochs=30, seed=SEED)
    print(f"seed {SEED}, val loss {vloss:.4f}\n")

    for sid in (4, 7):
        for ci, (c, m) in enumerate(zip(subjects[sid], summs[sid])):
            if len(m["trans"]) == 0:
                continue
            xn = (c["x"] - stats["mean"]) / stats["std"]
            ecp, emt = evidence_curves(model, xn)
            print(f"--- subject {sid} chunk {ci}: {len(m['trans'])} GT, "
                  f"{m['valid'].sum()/50:.0f} min valid ---")
            diagnose("CP", ecp, m["trans"], m["valid"])
            diagnose("MTM", emt, m["trans"], m["valid"])
            diagnose("CP+MTM", robust_z(ecp) + robust_z(emt), m["trans"], m["valid"])
            diagnose("B0 deriv", derivative_curve(xn), m["trans"], m["valid"])


if __name__ == "__main__":
    main()
