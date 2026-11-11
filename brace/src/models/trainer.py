"""Dataset and training utilities for one-step PDE surrogate operators."""
from __future__ import annotations

import os
import time

import numpy as np
import torch
from torch.utils.data import Dataset


class TrajectoryPairs(Dataset):
    """(u_t, u_{t+1}) pairs from a set of trajectories.

    ``stride`` subsamples the *starting times* only; every pair still spans
    exactly one saved interval, so the learned operator has the same dt as at
    evaluation time.
    """

    def __init__(self, u: np.ndarray, stride: int = 1, augment: bool = False):
        # u: [n_traj, n_time, ...]
        self.u = torch.as_tensor(u, dtype=torch.float32)
        self.n_traj, self.n_time = u.shape[0], u.shape[1]
        self.pairs = [(i, t) for i in range(self.n_traj)
                      for t in range(0, self.n_time - 1, stride)]
        self.augment = augment
        self.n_space = u.shape[-1]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        traj, t = self.pairs[idx]
        x, y = self.u[traj, t], self.u[traj, t + 1]
        if self.augment:
            # The PDE is translation invariant on the periodic domain.
            s = int(torch.randint(0, self.n_space, (1,)).item())
            x = torch.roll(x, shifts=s, dims=-1)
            y = torch.roll(y, shifts=s, dims=-1)
        return x, y


def relative_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return ((pred - target) ** 2).mean() / ((target**2).mean() + 1e-8)


def train_surrogate(
    model: torch.nn.Module,
    u_train: np.ndarray,
    u_val: np.ndarray,
    epochs: int = 60,
    batch_size: int = 32,
    lr: float = 2e-3,
    weight_decay: float = 1e-5,
    rollout: int = 1,
    pair_stride: int = 1,
    augment: bool = False,
    val_max_traj: int = 20,
    seed: int = 0,
    log_path: str | None = None,
    device: str = "cpu",
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ds = TrajectoryPairs(u_train, stride=pair_stride, augment=augment)
    # rollout > 1: sample a random start and backprop through `rollout` steps
    n_starts = ds.n_traj * (ds.n_time - rollout)
    gen = torch.Generator().manual_seed(seed)
    history = []
    best_val = float("inf")
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n_starts, generator=gen)
        total, count = 0.0, 0
        for i in range(0, n_starts, batch_size):
            idx = perm[i:i + batch_size]
            traj = idx // (ds.n_time - rollout)
            start = idx % (ds.n_time - rollout)
            u = ds.u[traj, start].to(device)
            loss = 0.0
            pred = u
            for k in range(rollout):
                pred = model(pred)
                target = ds.u[traj, start + k + 1].to(device)
                loss = loss + relative_mse(pred, target)
            loss = loss / rollout
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach()) * len(idx)
            count += len(idx)
        sched.step()
        model.eval()
        with torch.no_grad():
            val = evaluate_rollout(model, u_val, device=device, max_traj=val_max_traj)
        history.append({"epoch": ep, "train_relmse": total / max(count, 1), "val_relmse": val["rel_l2"]})
        if val["rel_l2"] < best_val:
            best_val = val["rel_l2"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if log_path and (ep % 5 == 0 or ep == epochs - 1):
            with open(log_path, "a") as f:
                f.write(f"ep={ep} train={history[-1]['train_relmse']:.5f} val={val['rel_l2']:.5f} "
                        f"time={time.time() - t0:.1f}s\n")
    model.load_state_dict(best_state)
    return {"history": history, "best_val_rel_l2": best_val, "train_seconds": time.time() - t0}


@torch.no_grad()
def evaluate_rollout(model: torch.nn.Module, u: np.ndarray, device: str = "cpu",
                     max_traj: int | None = None) -> dict:
    """Autoregressive rollout from the first snapshot; returns per-trajectory metrics."""
    model.eval()
    uu = torch.as_tensor(u, dtype=torch.float32).to(device)
    if max_traj:
        uu = uu[:max_traj]
    pred = uu[:, 0]
    errs, final_errs, step_errs = [], [], []
    for t in range(uu.shape[1] - 1):
        pred = model(pred)
        target = uu[:, t + 1]
        num = ((pred - target) ** 2).sum(dim=tuple(range(1, pred.dim())))
        den = (target**2).sum(dim=tuple(range(1, target.dim()))) + 1e-12
        e = torch.sqrt(num / den)
        errs.append(e)
        step_errs.append(e)
    errs = torch.stack(errs, dim=1)  # [B, T-1]
    final = errs[:, -1]
    return {
        "rel_l2": float(errs.mean()),
        "final_rel_l2": float(final.mean()),
        "max_rel_l2": float(errs.max()),
        "per_traj_mean": errs.mean(dim=1).cpu().numpy(),
        "per_traj_final": final.cpu().numpy(),
        "per_step": errs.cpu().numpy(),
    }


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.99):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def update(self, model: torch.nn.Module) -> None:
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k] = v.detach().clone()

    def copy_to(self, model: torch.nn.Module) -> None:
        model.load_state_dict(self.shadow)


def save_checkpoint(path: str, model: torch.nn.Module, extra: dict | None = None) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "extra": extra or {}}, path)
