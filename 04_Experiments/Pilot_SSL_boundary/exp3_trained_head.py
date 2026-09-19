"""Phase 2 experiment 2: trained boundary head on frozen SSL features (OTAS-lite).

RESULTS.md next-step #2. A small MLP learns to score candidate boundaries from
two-sided summary statistics of frozen encoder frame embeddings:

  f(t) = [mean(Z_L), std(Z_L), mean(Z_R), std(Z_R), |meanL-meanR|, cos(meanL,meanR)]

Positives: candidate positions centered on cuts of TRAIN-subject concat streams.
Negatives: positions >= 1.5 s away from any cut. Frozen encoder from SSL
pretraining (same as all other evidence families). Threshold selected on VAL
subject stream (never test). Evaluated sliding on TEST concat streams.
"""
from __future__ import annotations

import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, r"C:\Work\OpenCode\PhDYasmeen\04_Experiments\Pilot_SSL_boundary")
import pamap2_stream as ps
import pilot_concat as pc
from evaluate_boundary import match_f1, peak_pick
from exp2_distributional import frame_embeddings, robust_z
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import average_precision_score
from train_ssl import train_ssl

SIDE, GAP, STRIDE = 50, 10, 5   # frames: 1.0 s side, 0.25 s gap/stride... (12.5 Hz grid)
FEAT_DIM = 5 * 128 + 1          # mL,sL,mR,sR,|mL-mR| (128 each) + cos (1)


def features_at(Z, g):
    L = Z[g - GAP - SIDE: g - GAP]
    R = Z[g + GAP: g + GAP + SIDE]
    if len(L) < SIDE or len(R) < SIDE:
        return None
    mL, sL = L.mean(0), L.std(0)
    mR, sR = R.mean(0), R.std(0)
    cos = (mL @ mR) / (np.linalg.norm(mL) * np.linalg.norm(mR) + 1e-9)
    return np.concatenate([mL, sL, mR, sR, np.abs(mL - mR), [1 - cos]])


def make_head_dataset(model, subjects, summs, stats, rng, n_neg_per_pos=2):
    """positives at cuts, negatives >=1.5 s away, from train subjects' concat streams."""
    X, y = [], []
    for sid in ps.TRAIN_SUBJ:
        for attempt in range(3):            # 3 different concat shuffles per subject
            x, gt = pc.build_concat(subjects[sid], summs[sid], stats, rng)
            xn = (x - stats["mean"]) / stats["std"]
            Z = frame_embeddings(model, xn)
            Tf = len(Z)
            gtf = gt // 4
            pos = set()
            for g0 in gtf:
                for off in range(-STRIDE, STRIDE + 1, 2):   # label tolerance +-0.4 s
                    pos.add(int(g0 + off))
            pos = [g for g in pos if GAP + SIDE <= g < Tf - GAP - SIDE]
            far = np.ones(Tf, dtype=bool)
            for g0 in gtf:
                far[max(0, g0 - 19):g0 + 19] = False        # 1.5 s exclusion
            cand = np.where(far)[0]
            cand = cand[(cand > GAP + SIDE) & (cand < Tf - GAP - SIDE)]
            neg = rng.choice(cand, size=min(len(pos) * n_neg_per_pos, len(cand)),
                             replace=False)
            for g in pos:
                f = features_at(Z, g)
                if f is not None:
                    X.append(f); y.append(1)
            for g in neg:
                f = features_at(Z, g)
                if f is not None:
                    X.append(f); y.append(0)
    return np.asarray(X, np.float32), np.asarray(y, np.int64)


def train_head(X, y, Xv, yv, seed, epochs=8, lr=1e-3):
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = nn.Sequential(nn.Linear(FEAT_DIM, 256), nn.GELU(), nn.Dropout(0.2),
                        nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 1)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    Xt = torch.as_tensor(X, device=dev)
    yt = torch.as_tensor(y.astype(np.float32), device=dev)
    Xvt = torch.as_tensor(Xv, device=dev)
    yvt = torch.as_tensor(yv.astype(np.float32), device=dev)
    best_v, best_state = float("inf"), None
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(Xt), device=dev)
        for i in range(0, len(Xt), 512):
            xb, yb = Xt[perm[i:i + 512]], yt[perm[i:i + 512]]
            opt.zero_grad()
            loss = lossf(net(xb).squeeze(-1), yb)
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            vl = lossf(net(Xvt).squeeze(-1), yvt).item()
        if vl < best_v:
            best_v, best_state = vl, {k: v.detach().clone()
                                      for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    net.eval()
    return net


@torch.no_grad()
def head_curve(net, Z):
    dev = next(net.parameters()).device
    Tf = len(Z)
    ev = np.zeros(Tf, dtype=np.float32)
    lo = GAP + SIDE
    for g in range(lo, Tf - lo):
        f = features_at(Z, g)
        if f is None:
            continue
        v = net(torch.as_tensor(f, device=dev)[None]).item()
        ev[g] = v
    return ev


def evaluate(ev_frames, gt, T, thr):
    out = np.repeat(ev_frames, 4)
    if len(out) < T:
        out = np.pad(out, (0, T - len(out)), mode="edge")
    ev = robust_z(gaussian_filter1d(out[:T], 5))
    nm = np.zeros(T, dtype=bool)
    for g in gt:
        nm[max(0, g - 25):g + 26] = True
    r = np.corrcoef(nm.astype(float), ev)[0, 1]
    ap = average_precision_score(nm.astype(int), ev)
    p, rc, f1, mae, tp = match_f1(peak_pick(ev, thr), gt, 25)
    print(f"  trained-head r={r:+.3f} AUPRC={ap:.3f} (chance {nm.mean():.3f}) "
          f"| F1={f1:.3f} P={p:.2f} R={rc:.2f} MAE={mae:.2f}s")
    return r, ap, f1


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

    Xtr, ytr = make_head_dataset(model, subjects, summs, stats, rng)
    print(f"head train set: {len(ytr)} samples, {ytr.mean():.1%} positive")
    # validation stream features for early stop + threshold
    vx, vgt = pc.build_concat(subjects[4], summs[4], stats, rng)
    vxn = (vx - stats["mean"]) / stats["std"]
    vZ = frame_embeddings(model, vxn)
    Xv, yv = [], []
    vgtf = vgt // 4
    for g in range(GAP + SIDE, len(vZ) - GAP - SIDE):
        near = any(abs(g - g0) <= 10 for g0 in vgtf)
        Xv.append(features_at(vZ, g))
        yv.append(1 if near else 0)
    Xv = np.asarray(Xv, np.float32); yv = np.asarray(yv, np.int64)
    net = train_head(Xtr, ytr, Xv, yv, seed)
    # threshold sweep on validation evidence
    vev = robust_z(gaussian_filter1d(np.repeat(head_curve(net, vZ), 4)[:len(vx)], 5))
    best = (None, -1)
    for pct in range(90, 100):
        t = np.percentile(vev, pct)
        _, _, f1, _, _ = match_f1(peak_pick(vev, t), vgt, 25)
        if f1 > best[1]:
            best = (t, f1)
    print(f"val: {len(vgt)} GT, thr={best[0]:.2f}z, val F1={best[1]:.3f}\n")

    for sid in ps.TEST_SUBJ:
        x, gt = pc.build_concat(subjects[sid], summs[sid], stats, rng)
        if len(gt) == 0:
            continue
        xn = (x - stats["mean"]) / stats["std"]
        Z = frame_embeddings(model, xn)
        print(f"--- S{sid}: {len(x)//50}s, {len(gt)} GT ---")
        evaluate(head_curve(net, Z), gt, len(x), best[0])


if __name__ == "__main__":
    main()
