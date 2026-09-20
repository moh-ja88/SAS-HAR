"""SSL pretraining loop for the boundary pilot (DESIGN.md section 4)."""
from __future__ import annotations

import numpy as np
import torch

from ssl_model import SSLModel


def train_ssl(train_x, val_x, epochs=30, batch=64, lr=3e-4, wd=1e-4,
              patience=5, seed=42, device=None, n_ch=18, win=200):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = SSLModel(n_ch=n_ch, win=win).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    tr = torch.as_tensor(train_x, dtype=torch.float32, device=device)
    va = torch.as_tensor(val_x, dtype=torch.float32, device=device)

    best_val, best_state, bad = float("inf"), None, 0
    n = tr.shape[0]
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for i in range(0, n, batch):
            xb = tr[perm[i:i + batch]]
            loss, _ = model(xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * xb.shape[0]
        sched.step()
        model.eval()
        with torch.no_grad():
            vl = 0.0
            for i in range(0, va.shape[0], 512):
                vl += model(va[i:i + 512])[0].item() * va[i:i + 512].shape[0]
            vl /= va.shape[0]
        if vl < best_val - 1e-5:
            best_val, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val
