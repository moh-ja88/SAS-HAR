"""SSL model for the boundary pilot: dw-separable encoder + CP (JEPA-lite) head + MTM head.

DESIGN.md section 3. Input [B,18,200] -> encoder [B,128,50] (stride 4).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

WIN = 200
N_CH = 18
D_MODEL = 128
CP_K = 4            # predict 4 frames ahead (0.32 s)
MASK_TOTAL = 60     # 30% of 200
MASK_SPANS = [(25, 45), (95, 115), (165, 185)]


class DSConvBlock(nn.Module):
    def __init__(self, c_in, c_out, stride):
        super().__init__()
        self.dw = nn.Conv1d(c_in, c_in, 3, stride=stride, padding=1, groups=c_in)
        self.pw = nn.Conv1d(c_in, c_out, 1)
        self.norm = nn.BatchNorm1d(c_out)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.pw(self.dw(x))))


class Encoder(nn.Module):
    """[B,n_ch,win] -> [B,128,win/4] (total stride 4)."""

    def __init__(self, c_in=N_CH):
        super().__init__()
        self.blocks = nn.Sequential(
            DSConvBlock(c_in, 32, 2),
            DSConvBlock(32, 64, 2),
            DSConvBlock(64, D_MODEL, 1),
            DSConvBlock(D_MODEL, D_MODEL, 1),
        )

    def forward(self, x):
        return self.blocks(x)


class CPHead(nn.Module):
    """Latent predictive head: predict z_{t+k} from z_t (stop-grad target)."""

    def __init__(self, d=D_MODEL, k=CP_K):
        super().__init__()
        self.k = k
        self.pred = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def forward(self, z):
        s, t = z[:, :, :-self.k], z[:, :, self.k:]
        p = self.pred(s.permute(0, 2, 1)).permute(0, 2, 1)
        loss = F.smooth_l1_loss(p, t.detach())
        with torch.no_grad():
            err = (p - t).pow(2).mean(dim=1)  # [B, T-k] evidence
        return loss, err


class MTMHead(nn.Module):
    """Decoder mirror -> reconstruction [B,n_ch,win]; loss on masked positions only."""

    def __init__(self, win=WIN, n_ch=N_CH):
        super().__init__()
        def up(c_in, c_out, stride):
            k = 4 if stride > 1 else 3
            return nn.Sequential(nn.ConvTranspose1d(c_in, c_out, k, stride=stride, padding=1),
                                 nn.BatchNorm1d(c_out), nn.GELU())
        self.dec = nn.Sequential(
            up(D_MODEL, D_MODEL, 1), up(D_MODEL, 64, 2),
            up(64, 32, 2), nn.Conv1d(32, n_ch, 3, padding=1))
        mask = torch.zeros(win, dtype=torch.bool)
        for a, b in MASK_SPANS:
            mask[a * win // WIN: b * win // WIN] = True
        self.register_buffer("mask", mask)

    def forward(self, z, x, shift=0):
        x_hat = self.dec(z)
        m = self.mask
        if shift:
            m = torch.roll(m, shifts=shift)
        m = m[None, None, :]  # [1,1,200]
        loss = F.mse_loss(x_hat * m, x * m)
        with torch.no_grad():
            err = (x_hat - x).pow(2).mean(dim=1)  # [B,200]
            err = err * m.squeeze(1)
        return loss, err


class SSLModel(nn.Module):
    def __init__(self, n_ch=N_CH, win=WIN):
        super().__init__()
        self.encoder = Encoder(n_ch)
        self.cp = CPHead()
        self.mtm = MTMHead(win=win, n_ch=n_ch)

    def forward(self, x):
        z = self.encoder(x)
        loss_cp, err_cp = self.cp(z)
        loss_mtm, err_mtm = self.mtm(z, x)
        return loss_cp + loss_mtm, dict(cp=err_cp, mtm=err_mtm)


@torch.no_grad()
def evidence_curves(model, x, win=WIN, hop=50, device=None):
    """Sliding evidence over one continuous chunk x [T,18] (normalized).

    Returns (eps_cp[T], eps_mtm[T]): per-sample evidence, MAX over the
    overlapping windows that cover each sample (spike-preserving).
    CP: frame-rate evidence (T'-k frames) upsampled x4 to window positions.
    MTM: per-window SHIFTED mask (rotates coverage across windows); the
    returned evidence stays at its true sample position (no displacement).
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    T = x.shape[0]
    eps_cp = np.full(T, -np.inf)
    eps_mt = np.full(T, -np.inf)
    n_win = max(0, (T - win) // hop + 1)
    for i in range(n_win):
        s = i * hop
        w = torch.as_tensor(x[s:s + win].T, dtype=torch.float32, device=device)[None]
        z = model.encoder(w)
        _, err_cp = model.cp(z)
        _, err_mt = model.mtm(z, w, shift=(i * 17) % win)
        cp_up = err_cp.squeeze(0).cpu().repeat_interleave(4)
        n = s + cp_up.numel()
        eps_cp[s:n] = np.maximum(eps_cp[s:n], cp_up.numpy())
        eps_mt[s:s + win] = np.maximum(eps_mt[s:s + win], err_mt.squeeze(0).cpu().numpy())
    for arr in (eps_cp, eps_mt):
        bad = ~np.isfinite(arr)
        if bad.any():
            arr[bad] = np.median(arr[~bad])
    return eps_cp, eps_mt
