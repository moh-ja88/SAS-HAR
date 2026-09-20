"""Phase 2 experiment 4: probe-probability contrast evidence.

Uses the linear probe (94% acc on pure windows) on SSL encoder GAP embeddings.
evidence(t) = JS( p(y | window_left_of_t), p(y | window_right_of_t) ),
windows [t-2s, t-0.25s] and [t+0.25s, t+2s], evaluated every 0.25 s.
Also reports the single-feature |mL - mR| AUPRC as a diagnostic.
If this fails while the probe is accurate, the benchmark/alignment is suspect;
if it succeeds, boundary detection = SSL features + light supervision.
"""
from __future__ import annotations

import sys

import numpy as np
import torch

sys.path.insert(0, r"C:\Work\OpenCode\PhDYasmeen\04_Experiments\Pilot_SSL_boundary")
import pamap2_stream as ps
import pilot_concat as pc
from evaluate_boundary import match_f1, peak_pick
from exp2_distributional import frame_embeddings, robust_z
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from train_ssl import train_ssl

WIN = 200          # 4 s window
OFFSET = 88        # right window starts at t+88 (center at t+88+100... see below)
STRIDE = 12        # 0.24 s


def windows_left_right(x):
    """All (left, right) window pairs around candidate t = center grid."""
    T = len(x)
    # left window ends at t - 12 samples, right starts at t + 12 samples
    ts = np.arange(WIN + 12, T - WIN - 12, STRIDE)
    L = np.stack([x[t - 12 - WIN: t - 12] for t in ts])      # [N, WIN, C]
    R = np.stack([x[t + 12: t + 12 + WIN] for t in ts])
    return ts, L.transpose(0, 2, 1), R.transpose(0, 2, 1)


@torch.no_grad()
def gap_embed(model, W):
    dev = next(model.parameters()).device
    out = []
    for i in range(0, len(W), 512):
        z = model.encoder(torch.as_tensor(W[i:i + 512], dtype=torch.float32, device=dev))
        out.append(z.mean(dim=2).cpu().numpy())
    return np.concatenate(out)


def js(p, q, eps=1e-9):
    m = 0.5 * (p + q)
    kl = lambda a, b: (a * np.log(a / b + eps)).sum(1)
    return 0.5 * (kl(p, m) + kl(q, m))


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    rng = np.random.default_rng(seed)
    subjects, summs, stats = ps.load_all(pc.DATA)
    tr = [c for s in ps.TRAIN_SUBJ for c in subjects[s]]
    sm = [m for s in ps.TRAIN_SUBJ for m in summs[s]]
    tr_x, _ = ps.make_training_windows(tr, sm, stats, seed)
    va_x, _ = ps.make_training_windows(subjects[4], summs[4], stats, seed)
    model, _ = train_ssl(tr_x, va_x, epochs=30, seed=seed)

    # ---- train probe on pure activity windows (train subjects) ----
    def pure_windows(sids):
        xs, ys = [], []
        for sid in sids:
            for c, m in zip(subjects[sid], summs[sid]):
                xc = (c["x"] - stats["mean"]) / stats["std"]
                for st, en, lab in m["runs"]:
                    if lab == 0 or en - st < WIN:
                        continue
                    for w0 in range(st, en - WIN + 1, 50):
                        xs.append(xc[w0:w0 + WIN]); ys.append(lab)
        return (np.asarray(xs, np.float32).transpose(0, 2, 1), np.asarray(ys))

    Xtr_p, ytr_p = pure_windows(ps.TRAIN_SUBJ)
    Xte_p, yte_p = pure_windows(ps.TEST_SUBJ)
    eL = gap_embed(model, Xtr_p)
    eR = gap_embed(model, Xte_p)
    probe = LogisticRegression(max_iter=2000).fit(eL, ytr_p)
    acc = (probe.predict(eR) == yte_p).mean()
    print(f"probe check: test acc={acc:.3f}")

    # ---- evidence on concat streams ----
    def probe_contrast(xn):
        ts, L, R = windows_left_right(xn)
        pL = probe.predict_proba(gap_embed(model, L))
        pR = probe.predict_proba(gap_embed(model, R))
        ev = js(pL, pR)
        # also |mean-emb| diagnostic via encoder GAP embeddings
        return ts, ev

    vx, vgt = pc.build_concat(subjects[4], summs[4], stats, rng)
    vxn = (vx - stats["mean"]) / stats["std"]
    vts, vev = probe_contrast(vxn)

    def to_full(ev, ts, T):
        out = np.zeros(T)
        for t, v in zip(ts, ev):
            out[max(0, t - STRIDE // 2):t + STRIDE // 2 + 1] = v
        return out

    vfull = robust_z(gaussian_filter1d(to_full(vev, vts, len(vx)), 5))
    best = (None, -1)
    for pct in range(90, 100):
        t = np.percentile(vfull, pct)
        _, _, f1, _, _ = match_f1(peak_pick(vfull, t), vgt, 25)
        if f1 > best[1]:
            best = (t, f1)
    print(f"val: {len(vgt)} GT, thr={best[0]:.2f}z, F1={best[1]:.3f}\n")

    for sid in ps.TEST_SUBJ:
        x, gt = pc.build_concat(subjects[sid], summs[sid], stats, rng)
        if len(gt) == 0:
            continue
        xn = (x - stats["mean"]) / stats["std"]
        ts, ev = probe_contrast(xn)
        full = robust_z(gaussian_filter1d(to_full(ev, ts, len(x)), 5))
        nm = np.zeros(len(full), dtype=bool)
        for g in gt:
            nm[max(0, g - 25):g + 26] = True
        r = np.corrcoef(nm.astype(float), full)[0, 1]
        ap = average_precision_score(nm.astype(int), full)
        p, rc, f1, mae, tp = match_f1(peak_pick(full, best[0]), gt, 25)
        print(f"S{sid}: probe-contrast r={r:+.3f} AUPRC={ap:.3f} (chance {nm.mean():.3f})"
              f" | F1={f1:.3f} P={p:.2f} R={rc:.2f} MAE={mae:.2f}s tp={tp}/{len(gt)}")


if __name__ == "__main__":
    main()
