"""Boundary evaluation: statistics, tolerance-F1, NMS peak-picking, baselines.

Implements DESIGN.md section 5. All rates are in samples @ 50 Hz:
tolerance 25 samples (0.5 s), NMS radius 25, near-window 25, far-distance 50.
"""
from __future__ import annotations

import numpy as np
from scipy import stats as sps

TOL = 25          # 0.5 s matching tolerance
NMS = 25          # min separation between detections
NEAR = 25         # samples counted 'near' a boundary
FAR = 50          # min distance to count 'far'


def peak_pick(eps: np.ndarray, thr: float, nms: int = NMS) -> np.ndarray:
    """Threshold + run-grouping + argmax NMS. Returns detection indices."""
    above = eps > thr
    dets = []
    i = 0
    while i < len(above):
        if above[i]:
            j = i
            while j + 1 < len(above) and above[j + 1]:
                j += 1
            seg = eps[i:j + 1]
            dets.append(i + int(np.argmax(seg)))
            i = j + nms
        else:
            i += 1
    return np.asarray(dets, dtype=int)


def match_f1(dets: np.ndarray, gt: np.ndarray, tol: int = TOL):
    """One-to-one greedy matching. Returns (p, r, f1, mae_s, tp)."""
    if len(gt) == 0:
        return (0.0, 0.0, 0.0, float("nan"), 0)
    used = np.zeros(len(gt), dtype=bool)
    tp, errs = 0, []
    for d in dets:
        cand = np.where((np.abs(gt - d) <= tol) & ~used)[0]
        if len(cand):
            g = cand[np.argmin(np.abs(gt[cand] - d))]
            used[g] = True
            tp += 1
            errs.append(abs(gt[g] - d))
    p = tp / len(dets) if len(dets) else 0.0
    r = tp / len(gt)
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    mae = float(np.mean(errs)) / 50.0 if errs else float("nan")   # -> seconds
    return (p, r, f1, mae, tp)


def best_threshold(eps: np.ndarray, gt: np.ndarray, valid: np.ndarray):
    """Sweep percentiles on (validation) evidence; return best-F1 threshold."""
    e = eps[valid]
    best = (None, -1.0)
    for pct in range(60, 99):
        thr = np.percentile(e, pct)
        dets = peak_pick(eps * valid, thr)
        _, _, f1, _, _ = match_f1(dets, gt)
        if f1 > best[1]:
            best = (thr, f1)
    return best


def boundary_stats(eps: np.ndarray, gt: np.ndarray, valid: np.ndarray):
    """Point-biserial r + rank-sum test near vs far (valid samples only)."""
    near = np.zeros(len(eps), dtype=bool)
    for g in gt:
        near[max(0, g - NEAR):g + NEAR + 1] = True
    far = valid & ~near
    near = valid & near
    if near.sum() < 10 or far.sum() < 10:
        return dict(r=np.nan, p=np.nan)
    r, p = sps.pointbiserialr(near.astype(int)[near | far], eps[near | far])
    u = sps.mannwhitneyu(eps[near], eps[far], alternative="greater")
    return dict(r=float(r), p=float(min(p, u.pvalue)))


def derivative_curve(x_norm: np.ndarray) -> np.ndarray:
    """Baseline B0: L2 norm of first difference on normalized channels."""
    d = np.linalg.norm(np.diff(x_norm, axis=0), axis=1)
    return np.concatenate([[d[0]], d])


def chance_f1(gt: np.ndarray, n: int, n_dets: int, runs: int = 20, seed: int = 0):
    """Baseline B1: random detections at matched density."""
    rng = np.random.default_rng(seed)
    f1s = []
    for _ in range(runs):
        dets = rng.choice(n, size=min(n_dets, n), replace=False)
        _, _, f1, _, _ = match_f1(np.sort(dets), gt)
        f1s.append(f1)
    return float(np.mean(f1s))


def evaluate_curve(eps: np.ndarray, gt: np.ndarray, valid: np.ndarray, thr: float):
    dets = peak_pick(eps * valid, thr)
    p, r, f1, mae, tp = match_f1(dets, gt)
    st = boundary_stats(eps, gt, valid)
    return dict(thr=float(thr), precision=p, recall=r, f1=f1, me_s=mae,
                n_det=int(len(dets)), n_gt=int(len(gt)), **st)
