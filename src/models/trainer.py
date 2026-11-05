"""Trainer for the neural surrogate.

Builds (u_t, params, t) -> u_{t+dt} training pairs from trajectory files and
runs a standard MSE training loop on CPU with small batches.
"""

from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from .surrogate import NeuralSurrogate, SurrogateConfig


class TrajectoryPairsDataset(Dataset):
    """Samples (u_t, params, t) -> u_{t+dt} from trajectory .npz files.

    Each sample: u_t (nx,), params (n_params,) e.g. [nu], t (scalar), u_next (nx,).
    """

    def __init__(self, npz_paths: List[str], stride: int = 1, n_params: int = 1):
        self.samples: List[Tuple[np.ndarray, np.ndarray, float, np.ndarray]] = []
        import json
        for path in npz_paths:
            data = np.load(path)
            states = data["states"].astype(np.float64)  # (n_steps+1, nx)
            times = data["times"].astype(np.float64)
            data.close()
            meta_path = path[:-4] + ".json"
            meta = json.load(open(meta_path))
            nu = float(meta.get("nu", 0.01))
            params = np.array([nu], dtype=np.float64)
            for i in range(0, states.shape[0] - 1, stride):
                self.samples.append((states[i], params, float(times[i]), states[i + 1]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        return self.samples[idx]


def collate_pairs(batch):
    u_t = np.stack([b[0] for b in batch])
    p = np.stack([b[1] for b in batch])
    t = np.array([b[2] for b in batch], dtype=np.float64)
    u_next = np.stack([b[3] for b in batch])
    return u_t, p, t, u_next


def collect_npz_paths(directory: str) -> List[str]:
    paths = []
    for name in sorted(os.listdir(directory)):
        if name.endswith(".npz"):
            paths.append(os.path.join(directory, name))
    return paths


def train_surrogate(
    surrogate: NeuralSurrogate,
    train_dir: str,
    val_dir: str | None = None,
    stride: int = 1,
    device: str = "cpu",
) -> dict:
    """Train the surrogate on trajectory pairs from train_dir.

    Physical params (nu) and time t are derived from each trajectory's metadata.
    For simplicity here, nu is taken per-trajectory (constant in time) and t is
    the current time of the pair.
    """
    import json

    cfg = surrogate.config
    cfg.device = device
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    train_paths = collect_npz_paths(train_dir)
    # Build normalization stats from a subset of training states
    all_u = []
    all_p = []
    all_t = []
    for path in train_paths:
        data = np.load(path)
        states = data["states"].astype(np.float64)
        times = data["times"].astype(np.float64)
        data.close()
        meta_path = path[:-4] + ".json"
        meta = json.load(open(meta_path))
        nu = meta.get("nu", 0.01)
        # Sample a few states per trajectory for stats
        idx = np.linspace(0, states.shape[0] - 1, min(8, states.shape[0])).astype(int)
        all_u.append(states[idx])
        all_p.append(np.full((len(idx), 1), nu, dtype=np.float64))
        all_t.append(times[idx])
    u_all = np.concatenate(all_u, axis=0)
    p_all = np.concatenate(all_p, axis=0)
    t_all = np.concatenate(all_t, axis=0)
    surrogate.fit_normalization(u_all, p_all, t_all)

    # ---- increment (residual) normalization stats from consecutive pairs ----
    rng_inc = np.random.default_rng(cfg.seed + 1)
    inc_pairs = []
    step = max(1, len(train_paths) // 10)
    for path in train_paths[::step]:
        data = np.load(path)
        states = data["states"].astype(np.float64)
        data.close()
        n_pairs = min(16, states.shape[0] - 1)
        if n_pairs < 1:
            continue
        idx = rng_inc.choice(states.shape[0] - 1, size=n_pairs, replace=False)
        for i in idx:
            inc_pairs.append((states[i], states[i + 1]))
    if inc_pairs:
        surrogate.fit_increment_normalization(inc_pairs)

    train_ds = TrajectoryPairsDataset(train_paths, stride=stride, n_params=surrogate.config.n_params)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              collate_fn=collate_pairs, num_workers=0)

    model = surrogate.model
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    best_state = None

    val_loader = None
    if val_dir and os.path.isdir(val_dir):
        val_paths = collect_npz_paths(val_dir)
        val_ds = TrajectoryPairsDataset(val_paths, stride=stride, n_params=surrogate.config.n_params)
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                                collate_fn=collate_pairs, num_workers=0)

    inc_mean_t = torch.from_numpy(surrogate.inc_mean.astype(np.float32)).to(device)
    inc_std_t = torch.from_numpy(surrogate.inc_std.astype(np.float32)).to(device)

    model.train()
    for epoch in range(cfg.epochs):
        running = 0.0
        n_batches = 0
        for u_t, p, t, u_next in train_loader:
            un, pn, tn = surrogate._normalize(u_t, p, t)
            un = un.to(device); pn = pn.to(device); tn = tn.to(device).unsqueeze(-1)
            # target is the increment u_{t+dt} - u_t
            target = torch.from_numpy((u_next - u_t).astype(np.float32)).to(device)
            target_n = (target - inc_mean_t) / inc_std_t

            optimizer.zero_grad()
            out = model(un, pn, tn)
            loss = loss_fn(out, target_n)
            loss.backward()
            optimizer.step()
            running += float(loss.item())
            n_batches += 1
        train_loss = running / max(n_batches, 1)
        history["train_loss"].append(train_loss)

        val_loss = 0.0
        if val_loader is not None:
            model.eval()
            vrunning = 0.0; vn = 0
            with torch.no_grad():
                for u_t, p, t, u_next in val_loader:
                    un, pn, tn = surrogate._normalize(u_t, p, t)
                    un = un.to(device); pn = pn.to(device); tn = tn.to(device).unsqueeze(-1)
                    target = torch.from_numpy((u_next - u_t).astype(np.float32)).to(device)
                    target_n = (target - inc_mean_t) / inc_std_t
                    out = model(un, pn, tn)
                    vrunning += float(loss_fn(out, target_n).item())
                    vn += 1
            val_loss = vrunning / max(vn, 1)
            model.train()
        history["val_loss"].append(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


if __name__ == "__main__":
    print("Trainer module. Use train_surrogate(...) from an experiment script.")
