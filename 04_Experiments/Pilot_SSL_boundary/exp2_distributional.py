"""Phase 2 experiments: distributional contrast (MMD) + multi-scale sweep.

RESULTS.md next-step #1 and #3. Evidence computed on frame embeddings of the
frozen SSL encoder (precomputed once per stream at stride 4 = 12.5 Hz).

  MMD-RBF:  ev(t) = MMD^2_u(Z_L, Z_R), bandwidth = median heuristic (pooled)
  scales:   side w in {0.5, 1, 2} s, gap a in {0.25, 0.5} s
"""
from __future__ import annotations

import sys

import numpy as np
import torch

sys.path.insert(0, r"C:\Work\OpenCode\PhDYasmeen\04_Experiments\Pilot_SSL_boundary")
import pamap2_stream as ps
import pilot_concat as pc
from evaluate_boundary import match_f1, peak_pick
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import average_precision_score
from train_ssl import train_ssl


@torch.no_grad()
def frame_embeddings(model, x, win=200, hop=50):
    """[Tf, D] frame embeddings (stride-4 grid); uncovered tails backfilled."""
    dev = next(model.parameters()).device
    T = len(x)
    Tf = (T - win) // 4 + 2
    first = model.encoder(torch.as_tensor(
        x[0:win].T, dtype=torch.float32, device=dev)[None]).squeeze(0)[:, 0]
    Z = np.tile(first.cpu().numpy()[:, None].T, (Tf, 1)).astype(np.float32)
    for i in range((T - win) // hop + 1):
        s = i * hop
        w = torch.as_tensor(x[s:s + win].T, dtype=torch.float32, device=dev)[None]
        z = model.encoder(w).squeeze(0)
        g = s // 4
        n = min(z.shape[1], Tf - g)
        if n > 0:
            Z[g:g + n] = z[:, :n].cpu().numpy().T
    return Z


def mmd2_rbf(A: np.ndarray, B: np.ndarray) -> float:
    """Unbiased MMD^2 with RBF kernel, median-heuristic bandwidth."""
    X = np.concatenate([A, B])
    d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    sig2 = np.median(d2[d2 > 0]) + 1e-9
    K = np.exp(-d2 / sig2)
    n, m = len(A), len(B)
    kxx = (K[:n, :n].sum() - np.trace(K[:n, :n])) / (n * (n - 1))
    kyy = (K[n:, n:].sum() - np.trace(K[n:, n:])) / (m * (m - 1))
    kxy = K[:n, n:].mean()
    return float(kxx + kyy - 2 * kxy)


def mmd_curve(Z, fs_frames=4, side_s=1.0, gap_s=0.25, stride_s=0.25):
    """MMD evidence on a (Tf,) grid at stride_s resolution; upsampled to frames."""
    side = int(side_s * fs_frames * 4) // 4      # frames per side
    gap = max(1, int(gap_s * fs_frames * 4) // 4)
    stride = max(1, int(stride_s * fs_frames * 4) // 4)
    Tf = len(Z)
    ev = np.zeros(Tf, dtype=np.float32)
    lo = gap + side
    for g in range(lo, Tf - lo, stride):
        L = Z[g - gap - side: g - gap]
        R = Z[g + gap: g + gap + side]
        v = mmd2_rbf(L, R)
        ev[g - stride // 2: g + stride // 2 + 1] = v
    return ev


def robust_z(e):
    med = np.median(e)
    mad = np.median(np.abs(e - med)) + 1e-9
    return (e - med) / (1.4826 * mad)


def to_samples(ev_frames, T):
    """frame grid -> sample grid (repeat 4), edge-padded to exact T."""
    out = np.repeat(ev_frames, 4)
    if len(out) < T:
        out = np.pad(out, (0, T - len(out)), mode="edge")
    return out[:T]


def evaluate(name, ev, gt, T, thr=None):
    ev = robust_z(gaussian_filter1d(ev, 5))
    nm = np.zeros(T, dtype=bool)
    for g in gt:
        nm[max(0, g - 25):g + 26] = True
    r = np.corrcoef(nm.astype(float), ev)[0, 1]
    ap = average_precision_score(nm.astype(int), ev)
    line = f"  {name:28s} r={r:+.3f} AUPRC={ap:.3f} (chance {nm.mean():.3f})"
    if thr is not None:
        p, rc, f1, mae, tp = match_f1(peak_pick(ev, thr), gt, 25)
        line += f" | F1={f1:.3f} P={p:.2f} R={rc:.2f} MAE={mae:.2f}s"
    print(line)
    return r, ap


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    rng = np.random.default_rng(seed)
    subjects, summs, stats = ps.load_all(pc.DATA)
    tr = [c for s in ps.TRAIN_SUBJ for c in subjects[s]]
    sm = [m for s in ps.TRAIN_SUBJ for m in summs[s]]
    tr_x, _ = ps.make_training_windows(tr, sm, stats, seed)
    va_x, _ = ps.make_training_windows(subjects[4], summs[4], stats, seed)
    model, _ = train_ssl(tr_x, va_x, epochs=30, seed=seed)

    # validation stream: embeddings + scale sweep for threshold selection
    vx, vgt = pc.build_concat(subjects[4], summs[4], stats, rng)
    vxn = (vx - stats["mean"]) / stats["std"]
    vZ = frame_embeddings(model, vxn)

    scales = [(0.5, 0.25), (1.0, 0.25), (2.0, 0.25), (1.0, 0.5), (2.0, 0.5)]
    best = {}
    for side, gap in scales:
        ev = to_samples(mmd_curve(vZ, side_s=side, gap_s=gap), len(vx))
        evz = robust_z(gaussian_filter1d(ev, 5))
        b = (None, -1)
        for pct in range(90, 100):
            t = np.percentile(evz, pct)
            _, _, f1, _, _ = match_f1(peak_pick(evz, t), vgt, 25)
            if f1 > b[1]:
                b = (t, f1)
        best[(side, gap)] = b
        print(f"val scale side={side}s gap={gap}s: thr={b[0]:.2f}z F1={b[1]:.3f}")
    top_scale = max(best, key=lambda k: best[k][1])
    print(f"best val scale: side={top_scale[0]}s gap={top_scale[1]}s\n")

    for sid in ps.TEST_SUBJ:
        x, gt = pc.build_concat(subjects[sid], summs[sid], stats, rng)
        if len(gt) == 0:
            continue
        xn = (x - stats["mean"]) / stats["std"]
        Z = frame_embeddings(model, xn)
        print(f"--- S{sid}: {len(x)//50}s, {len(gt)} GT ---")
        for side, gap in scales:
            ev = to_samples(mmd_curve(Z, side_s=side, gap_s=gap), len(x))
            evaluate(f"MMD side={side}s gap={gap}s", ev, gt, len(x),
                     best[(side, gap)][0] if (side, gap) == top_scale else None)


if __name__ == "__main__":
    main()
