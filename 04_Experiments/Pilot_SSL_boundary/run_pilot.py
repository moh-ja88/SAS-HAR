"""End-to-end pilot driver (DESIGN.md sections 4-8).

Usage (system Python 3.13 with torch 2.6):
  python run_pilot.py [--data DATA_ROOT] [--epochs N] [--seeds 42 43 44]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import pamap2_stream as ps
from evaluate_boundary import (best_threshold, chance_f1, derivative_curve,
                               evaluate_curve)
from ssl_model import evidence_curves
from train_ssl import train_ssl

HERE = Path(__file__).parent
DATA_ROOT = (HERE / ".." / ".." / "05_Code" / "PhD-HAR-Segmentation" / "data"
             / "pamap2" / "PAMAP2_Dataset" / "Protocol").resolve()


def znorm(x, stats):
    return (x - stats["mean"]) / stats["std"]


def run_seed(seed, subjects, summs, stats, epochs, device, out_dir):
    print(f"\n=== seed {seed} ===")
    t0 = time.time()

    tr_x, _ = ps.make_training_windows(
        [c for s in ps.TRAIN_SUBJ for c in subjects[s]],
        [m for s in ps.TRAIN_SUBJ for m in summs[s]], stats, seed)
    va_x, _ = ps.make_training_windows(subjects[4], summs[4], stats, seed)
    print(f"windows: train={len(tr_x)} val={len(va_x)}")

    model, vloss = train_ssl(tr_x, va_x, epochs=epochs, seed=seed, device=device)
    print(f"trained in {time.time()-t0:.0f}s, best val loss {vloss:.4f}")

    # evidence + threshold selection on validation subject
    val_eps = {"cp": [], "mtm": []}
    val_gt, val_valid = [], []
    for c, m in zip(subjects[4], summs[4]):
        if len(m["trans"]) == 0:      # chunk without usable GT
            continue
        ecp, emt = evidence_curves(model, znorm(c["x"], stats), device=device)
        val_eps["cp"].append(ecp); val_eps["mtm"].append(emt)
        val_gt.append(m["trans"]); val_valid.append(m["valid"])
    thr = {}
    for head in ("cp", "mtm"):
        fs = []
        for e, g, v in zip(val_eps[head], val_gt, val_valid):
            t, f = best_threshold(e, g, v)
            fs.append(f)
        # pick threshold from the chunk with most GT (simple, honest)
        gi = int(np.argmax([len(g) for g in val_gt]))
        thr[head], _ = best_threshold(val_eps[head][gi], val_gt[gi], val_valid[gi])
        print(f"val F1[{head}] ~ {np.mean(fs):.3f}, thr={thr[head]:.4g}")

    results = {"seed": seed, "val_loss": vloss, "thresholds": thr, "subjects": {}}
    for sid in ps.TEST_SUBJ:
        subj = {}
        for head in ("cp", "mtm"):
            per_chunk = []
            for ci, (c, m) in enumerate(zip(subjects[sid], summs[sid])):
                if len(m["trans"]) == 0:   # no GT in chunk (e.g. subject109 NaN gaps)
                    continue
                xn = znorm(c["x"], stats)
                ecp, emt = evidence_curves(model, xn, device=device)
                eps = ecp if head == "cp" else emt
                r = evaluate_curve(eps, m["trans"], m["valid"], thr[head])
                # baselines on the same chunk
                b0 = evaluate_curve(derivative_curve(xn) * m["valid"],
                                    m["trans"], m["valid"],
                                    best_threshold(derivative_curve(xn) * m["valid"],
                                                   m["trans"], m["valid"])[0])
                b1 = chance_f1(m["trans"], len(eps), max(r["n_det"], 1), seed=seed + ci)
                r.update(b0_f1=b0["f1"], chance_f1=b1)
                per_chunk.append(r)
                if head == "cp" and ci == 0 and len(m["trans"]) > 3:
                    plot_chunk(out_dir, seed, sid, xn, ecp, emt, m)
            if not per_chunk:
                subj[head] = dict(chunks=[], f1=float("nan"), r=float("nan"),
                                  me_s=float("nan"), b0_f1=float("nan"),
                                  chance_f1=float("nan"), n_gt=0)
                continue
            agg = lambda k: float(np.nanmean([p[k] for p in per_chunk]))
            subj[head] = dict(chunks=per_chunk, f1=agg("f1"), r=agg("r"),
                              me_s=agg("me_s"), b0_f1=agg("b0_f1"),
                              chance_f1=agg("chance_f1"), n_gt=sum(p["n_gt"] for p in per_chunk))
        results["subjects"][sid] = subj
        for h in ("cp", "mtm"):
            s = subj[h]
            print(f"S{sid} {h.upper():3s}: F1={s['f1']:.3f} (B0={s['b0_f1']:.3f}, "
                  f"chance={s['chance_f1']:.3f}) r={s['r']:.2f} MAE={s['me_s']:.2f}s")
    return results


def plot_chunk(out_dir, seed, sid, xn, ecp, emt, m, t0=0, span=3000):
    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    tt = np.arange(span) / 50.0
    for lab, e in (("CP (JEPA-lite)", ecp), ("MTM (recon)", emt)):
        a = ax[0] if lab.startswith("CP") else ax[1]
        a.plot(tt, e[t0:t0 + span], lw=0.7)
        for g in m["trans"]:
            if t0 <= g < t0 + span:
                a.axvline((g - t0) / 50.0, color="r", ls="--", lw=0.8, alpha=0.7)
        a.set_title(f"{lab} — subject {sid} chunk0 (red = GT transitions)")
        a.set_ylabel("evidence")
    ax[1].set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig(out_dir / f"evidence_seed{seed}_S{sid}.png", dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA_ROOT))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    print(f"device={device}  data={args.data}")

    subjects, summs, stats = ps.load_all(args.data)
    for sid, ch in subjects.items():
        n_tr = sum(len(m["trans"]) for m in summs[sid])
        mins = sum(c["x"].shape[0] for c in ch) / 50 / 60
        print(f"subject {sid}: {len(ch)} chunks, {mins:.0f} min, {n_tr} transitions")

    all_results = []
    for seed in args.seeds:
        all_results.append(run_seed(seed, subjects, summs, stats,
                                    args.epochs, device, out_dir))
        (out_dir / f"results_seed{seed}.json").write_text(
            json.dumps(all_results[-1], indent=1, default=float))

    # cross-seed summary vs pre-registered criteria (DESIGN.md section 7)
    summary = {"per_seed": [], "criteria": {}}
    for r in all_results:
        for sid, subj in r["subjects"].items():
            for h in ("cp", "mtm"):
                summary["per_seed"].append(
                    dict(seed=r["seed"], subject=sid, head=h,
                         **{k: subj[h][k] for k in
                            ("f1", "b0_f1", "chance_f1", "r", "me_s", "n_gt")}))
    cp = [c for c in summary["per_seed"] if c["head"] == "cp"]
    summary["criteria"]["S2_cp_beats_B0_by_0.10"] = bool(
        cp and all(c["f1"] >= c["b0_f1"] + 0.10 for c in cp))
    summary["criteria"]["S3_MAE_under_0.5s"] = bool(
        cp and all(np.nanmean([c["me_s"] for c in cp if c["subject"] == s]) < 0.5
                   for s in {c["subject"] for c in cp}))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print("\n=== criteria ===")
    for k, v in summary["criteria"].items():
        print(f"{k}: {'PASS' if v else 'FAIL'}")


if __name__ == "__main__":
    main()
