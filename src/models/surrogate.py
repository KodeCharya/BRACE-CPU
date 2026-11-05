"""Surrogate wrapper: normalization + single-step predict + autoregressive rollout.

Keeps input/output normalization statistics computed from the training set so
the raw MLP sees normalized data while callers work in physical units.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import torch

from .mlp import MLPSurrogate, build_model


@dataclass
class SurrogateConfig:
    nx: int = 64
    n_params: int = 1      # e.g. [nu]
    n_time: int = 1        # [t]
    hidden_layers: int = 3
    hidden_units: int = 64
    activation: str = "relu"
    lr: float = 1e-3
    batch_size: int = 256
    epochs: int = 50
    seed: int = 0
    device: str = "cpu"
    dtype: str = "float32"


class NeuralSurrogate:
    def __init__(self, config: SurrogateConfig | None = None):
        self.config = config or SurrogateConfig()
        self.model = build_model(
            nx=self.config.nx,
            n_params=self.config.n_params,
            n_time=self.config.n_time,
            hidden_layers=self.config.hidden_layers,
            hidden_units=self.config.hidden_units,
            activation=self.config.activation,
        ).to(self.config.device)
        self.model.float()
        # Normalization statistics (set by fit_normalization / load)
        self.u_mean: np.ndarray | None = None
        self.u_std: np.ndarray | None = None
        self.p_mean: np.ndarray | None = None
        self.p_std: np.ndarray | None = None
        self.t_mean: float = 0.0
        self.t_std: float = 1.0
        # Increment (residual) prediction stats: model learns u_{t+dt} - u_t
        self.inc_mean: np.ndarray | None = None
        self.inc_std: np.ndarray | None = None

    # ------------------------------------------------------------------ #
    # Normalization
    # ------------------------------------------------------------------ #
    def fit_normalization(self, u_all: np.ndarray, p_all: np.ndarray, t_all: np.ndarray) -> None:
        """Compute normalization stats from arrays."""
        u_all = np.asarray(u_all, dtype=np.float64)
        p_all = np.asarray(p_all, dtype=np.float64)
        t_all = np.asarray(t_all, dtype=np.float64).reshape(-1)
        self.u_mean = u_mean = u_all.mean(axis=0)
        self.u_std = u_std = u_all.std(axis=0) + 1e-8
        self.p_mean = p_all.mean(axis=0)
        self.p_std = p_all.std(axis=0) + 1e-8
        self.t_mean = float(t_all.mean())
        self.t_std = float(t_all.std() + 1e-8)

    def fit_increment_normalization(self, pairs: list[tuple[np.ndarray, np.ndarray]]) -> None:
        """Compute normalization stats for the increment u_{t+dt} - u_t.

        ``pairs`` is a list of (u_t, u_next) arrays (subsampled is fine).
        """
        incs = np.stack([np.asarray(u_next, dtype=np.float64) - np.asarray(u_t, dtype=np.float64)
                         for u_t, u_next in pairs], axis=0)
        self.inc_mean = incs.mean(axis=0)
        self.inc_std = incs.std(axis=0) + 1e-8

    def _denormalize_increment(self, inc_n: np.ndarray) -> np.ndarray:
        assert self.inc_mean is not None, "increment normalization not fitted"
        return inc_n * self.inc_std + self.inc_mean

    def _normalize(self, u: np.ndarray, p: np.ndarray, t: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.u_mean is not None, "normalization not fitted"
        un = (u - self.u_mean) / self.u_std
        pn = (p - self.p_mean) / self.p_std
        tn = (t - self.t_mean) / self.t_std
        return (
            torch.from_numpy(un.astype(np.float32)),
            torch.from_numpy(pn.astype(np.float32)),
            torch.from_numpy(tn.astype(np.float32)),
        )

    def _denormalize(self, u_n: np.ndarray) -> np.ndarray:
        assert self.u_mean is not None
        return u_n * self.u_std + self.u_mean

    # ------------------------------------------------------------------ #
    # Single-step prediction
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def predict(self, u: np.ndarray, params: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Predict next state.

        u: (nx,) or (B, nx); params: (n_params,) or (B, n_params); t: (B,) or scalar.
        Returns same batch shape as u.
        """
        self.model.eval()
        u = np.asarray(u, dtype=np.float64)
        p = np.asarray(params, dtype=np.float64)
        t = np.asarray(t, dtype=np.float64)
        single = u.ndim == 1
        if single:
            u = u[None, :]
            p = p[None, :] if p.ndim == 1 else p
            t = np.array([float(t)]) if t.ndim == 0 else t.reshape(-1)
        B = u.shape[0]
        if p.ndim == 1:
            p = np.tile(p, (B, 1))
        if t.ndim == 0 or t.size == 1:
            t = np.full(B, float(t.reshape(-1)[0]))
        un, pn, tn = self._normalize(u, p, t)
        un = un.to(self.config.device)
        pn = pn.to(self.config.device)
        tn = tn.to(self.config.device).unsqueeze(-1)
        inc_n = self.model(un, pn, tn)
        inc = self._denormalize_increment(inc_n.cpu().numpy().astype(np.float64))
        u_next = u + inc
        if single:
            return u_next[0]
        return u_next

    # ------------------------------------------------------------------ #
    # Autoregressive rollout
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def rollout(self, u0: np.ndarray, params: np.ndarray, t0: float, dt: float, n_steps: int) -> np.ndarray:
        """Roll forward n_steps of size dt starting from u0.

        Returns array of shape (n_steps+1, nx) including u0.
        """
        self.model.eval()
        u0 = np.asarray(u0, dtype=np.float64)
        p = np.asarray(params, dtype=np.float64).reshape(-1)
        states = np.empty((n_steps + 1, u0.shape[0]), dtype=np.float64)
        states[0] = u0
        u = u0.copy()
        t = t0
        for i in range(n_steps):
            t_arr = np.full(1, t, dtype=np.float64)
            u = self.predict(u, p, t_arr)
            states[i + 1] = u
            t += dt
        return states

    def save(self, path: str) -> None:
        torch.save({
            "model_state": self.model.state_dict(),
            "config": self.config.__dict__,
            "u_mean": self.u_mean,
            "u_std": self.u_std,
            "p_mean": self.p_mean,
            "p_std": self.p_std,
            "t_mean": self.t_mean,
            "t_std": self.t_std,
            "inc_mean": self.inc_mean,
            "inc_std": self.inc_std,
        }, path)

    def load(self, path: str, device: str = "cpu") -> None:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        self.config = SurrogateConfig(**ckpt["config"])
        self.model = build_model(
            nx=self.config.nx, n_params=self.config.n_params, n_time=self.config.n_time,
            hidden_layers=self.config.hidden_layers, hidden_units=self.config.hidden_units,
            activation=self.config.activation,
        ).to(device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.float()
        self.u_mean = ckpt["u_mean"]
        self.u_std = ckpt["u_std"]
        self.p_mean = ckpt["p_mean"]
        self.p_std = ckpt["p_std"]
        self.t_mean = ckpt["t_mean"]
        self.t_std = ckpt["t_std"]
        self.inc_mean = ckpt.get("inc_mean", None)
        self.inc_std = ckpt.get("inc_std", None)
