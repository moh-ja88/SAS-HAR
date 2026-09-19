"""PAMAP2 continuous-stream loading for the SSL-boundary pilot.

Produces per-subject continuous chunks (50 Hz, 18 IMU channels), run-level labels,
transition ground truth, and subject-wise splits. See DESIGN.md section 2.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import resample_poly

FS_IN = 100          # raw PAMAP2 sampling rate
FS = 50              # pilot working rate
N_CH = 18            # 3 IMUs x (acc xyz + gyro xyz)
# 0-based column indices in the 54-col .dat files
COLS = [4, 5, 6, 10, 11, 12,        # hand  acc + gyro
        21, 22, 23, 27, 28, 29,     # chest acc + gyro
        38, 39, 40, 44, 45, 46]     # ankle acc + gyro
ACT_COL = 1
TIME_COL = 0
NULL_LABEL = 0
MIN_RUN_S = 1.0       # merge label runs shorter than this
GAP_S = 0.1           # time gap that splits a stream into chunks
WIN = 200             # 4 s @ 50 Hz
HOP_TRAIN = 50        # 1 s hop for training windows
HOP_EVAL = 50         # sliding hop at eval

TRAIN_SUBJ = [1, 2, 3, 5, 6, 8]
VAL_SUBJ = [4]
TEST_SUBJ = [7, 9]


def load_subject_chunks(path: Path) -> list[dict]:
    """Load one subject .dat -> list of continuous chunks at 50 Hz.

    Each chunk: dict(x[float, T, 18] un-normalized, label[int, T], time[float, T])
    Chunks split where dropped NaN rows create time gaps > GAP_S.
    """
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="c")
    arr = df.to_numpy(dtype=np.float64)
    x = arr[:, COLS]
    lab = arr[:, ACT_COL].astype(int)
    t = arr[:, TIME_COL]

    ok = ~np.isnan(x).any(axis=1)
    x, lab, t = x[ok], lab[ok], t[ok]

    chunks, start = [], 0
    dt_gaps = np.where(np.diff(t) > GAP_S * FS_IN)[0]
    edges = list(dt_gaps + 1) + [len(t)]
    for stop in edges:
        if stop - start < WIN:  # too short to be useful
            start = stop
            continue
        xs = resample_poly(x[start:stop], 1, 2, axis=0)
        ls = lab[start:stop][::2]
        ts = t[start:stop][::2]
        chunks.append(dict(x=xs, label=ls, time=ts))
        start = stop
    return chunks


def merge_short_runs(label: np.ndarray) -> np.ndarray:
    """Merge runs shorter than MIN_RUN_S into the previous run."""
    out = label.copy()
    change = np.where(np.diff(out) != 0)[0] + 1
    bounds = np.concatenate([[0], change, [len(out)]])
    for i in range(len(bounds) - 1):
        if (bounds[i + 1] - bounds[i]) < MIN_RUN_S * FS and i > 0:
            out[bounds[i]:bounds[i + 1]] = out[bounds[i] - 1]
    return out


def runs_of(label: np.ndarray) -> list[tuple[int, int, int]]:
    """(start, stop, label) runs."""
    change = np.where(np.diff(label) != 0)[0] + 1
    bounds = np.concatenate([[0], change, [len(label)]])
    return [(int(bounds[i]), int(bounds[i + 1]), int(label[bounds[i]]))
            for i in range(len(bounds) - 1)]


def transitions_from_runs(runs: list[tuple[int, int, int]]) -> np.ndarray:
    """GT transition positions: EDGES of valid (non-null) runs.

    PAMAP2 protocol sandwiches activities between long NULL pauses (30-250 s),
    so real postural changes occur where activities start/end — including the
    activity<->NULL edges (subject starts/stops moving), and direct edges
    between adjacent valid runs (e.g. stairs up->down). NULL midpoints are
    stillness, NOT transitions (verified empirically 2026-09-18).
    """
    out = set()
    for s, e, lab in runs:
        if lab == NULL_LABEL:
            continue
        out.add(s)   # entering the activity (from NULL or another activity)
        out.add(e)   # leaving the activity
    return np.asarray(sorted(out), dtype=int)


def valid_mask_of(runs: list[tuple[int, int, int]], T: int) -> np.ndarray:
    m = np.zeros(T, dtype=bool)
    for s, e, lab in runs:
        if lab != NULL_LABEL:
            m[s:e] = True
    return m


def subject_summary(chunks: list[dict]) -> list[dict]:
    """Runs + GT transitions + valid mask per chunk (labels merged first)."""
    out = []
    for c in chunks:
        lab = merge_short_runs(c["label"])
        runs = runs_of(lab)
        out.append(dict(runs=runs,
                        trans=transitions_from_runs(runs),
                        valid=valid_mask_of(runs, len(lab)),
                        label=lab))
    return out


def make_training_windows(chunks, summaries, stats, seed):
    """Pure windows (fully inside valid runs) from train/val subjects."""
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for c, s in zip(chunks, summaries):
        xc = (c["x"] - stats["mean"]) / stats["std"]
        for st, en, lab in s["runs"]:
            if lab == NULL_LABEL or en - st < WIN:
                continue
            for w0 in range(st, en - WIN + 1, HOP_TRAIN):
                xs.append(xc[w0:w0 + WIN])
                ys.append(lab)
    xs = np.asarray(xs, dtype=np.float32).transpose(0, 2, 1)   # [N, 18, WIN]
    idx = rng.permutation(len(xs))
    return xs[idx], np.asarray(ys)[idx]


def load_all(data_root: str):
    root = Path(data_root)
    subjects = {}
    for sid in TRAIN_SUBJ + VAL_SUBJ + TEST_SUBJ:
        p = root / f"subject{100 + sid}.dat"
        subjects[sid] = load_subject_chunks(p)
    # normalization stats from TRAIN subjects only
    pool = np.concatenate([c["x"] for sid in TRAIN_SUBJ for c in subjects[sid]])
    stats = dict(mean=pool.mean(axis=0), std=pool.std(axis=0) + 1e-8)
    summs = {sid: subject_summary(subjects[sid]) for sid in subjects}
    return subjects, summs, stats
