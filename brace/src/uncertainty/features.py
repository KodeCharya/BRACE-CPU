"""Risk features for a neural PDE surrogate rollout.

All features are computable online from the surrogate ensemble, the PDE
operator, and the prediction history. No ground-truth state is used.
"""
from __future__ import annotations

import numpy as np
import torch

FEATURE_NAMES = [
    "ens_std_mean",
    "ens_std_max",
    "residual_rel",
    "conservation_dev",
    "update_mag",
    "temporal_inconsistency",
    "hist_residual",
    "hist_std",
    "time_frac",
]

EPS = 1e-8


@torch.no_grad()
def ensemble_predict(models, u: np.ndarray, device: str = "cpu"):
    """Return ensemble predictions [M, N, *spatial] for batched states u."""
    x = torch.as_tensor(u, dtype=torch.float32, device=device)
    preds = [m(x).cpu().numpy() for m in models]
    return np.stack(preds, axis=0)


def rms(x: np.ndarray, axes=(1,)) -> np.ndarray:
    return np.sqrt(np.mean(x**2, axis=axes) + EPS)


def compute_features(
    u_prev: np.ndarray,
    u_cur: np.ndarray,
    preds: np.ndarray,
    solver,
    dt: float,
    history: dict,
    t: int,
    n_time: int,
    params: dict | None = None,
) -> np.ndarray:
    """Features of the ensemble prediction for the next step.

    Parameters
    ----------
    u_prev : state at t-1 (same for t=0 as u_cur)
    u_cur  : state at t
    preds  : [M, N, ...] ensemble predictions of u_{t+1}
    params : per-trajectory PDE coefficients (nu/cx/cy/eps) for residual
    """
    mean = preds.mean(axis=0)
    # Broadcast per-trajectory PDE coefficients against the field axes.
    bshape = (mean.shape[0],) + (1,) * (mean.ndim - 1)
    params = {k: np.asarray(v, dtype=np.float64).reshape(bshape) if np.ndim(v) > 0 else float(v)
              for k, v in (params or {}).items()}
    std = preds.std(axis=0)
    axes = tuple(range(1, mean.ndim))
    scale = rms(u_cur, axes=axes)
    feat = np.zeros((u_cur.shape[0], len(FEATURE_NAMES)), dtype=np.float64)
    feat[:, 0] = rms(std, axes=axes) / scale
    feat[:, 1] = std.reshape(std.shape[0], -1).max(axis=1) / scale
    res = solver.temporal_residual(u_cur, mean, dt, **params)
    feat[:, 2] = rms(res, axes=axes) / (rms(mean - u_cur, axes=axes) / dt + EPS)
    feat[:, 3] = np.abs(mean.mean(axis=axes) - u_cur.mean(axis=axes)) / scale
    feat[:, 4] = rms(mean - u_cur, axes=axes) / scale
    extrap = 2.0 * u_cur - u_prev
    feat[:, 5] = rms(mean - extrap, axes=axes) / scale if t > 0 else 0.0
    feat[:, 6] = history["residual"]
    feat[:, 7] = history["std"]
    feat[:, 8] = t / max(n_time - 1, 1)
    history["residual"] = 0.8 * history["residual"] + 0.2 * feat[:, 2]
    history["std"] = 0.8 * history["std"] + 0.2 * feat[:, 0]
    return feat


def new_history(n: int) -> dict:
    return {"residual": np.zeros(n), "std": np.zeros(n)}


def step_error(pred: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Per-trajectory relative L2 error of a single step."""
    axes = tuple(range(1, pred.ndim))
    num = np.sqrt(np.mean((pred - ref) ** 2, axis=axes))
    den = np.sqrt(np.mean(ref**2, axis=axes)) + EPS
    return num / den
