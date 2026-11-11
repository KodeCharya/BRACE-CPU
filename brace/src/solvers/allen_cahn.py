"""1D Allen-Cahn equation on a periodic domain.

    u_t = eps * u_xx + u - u^3,   x in [0, 2*pi)

Space: Fourier pseudo-spectral with 2/3 dealiasing.
Time:  Strang splitting: exact integrating factor for the linear diffusion
       term and a classical RK4 step for the nonlinear reaction term.
The reaction term is integrated explicitly, so dt is chosen well below the
reaction stiffness limit (|d/dU (u - u^3)| <= 2).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .spectral import dealias_mask_1d, wavenumbers_1d


@dataclass
class AllenCahnConfig:
    n: int = 128
    eps: float = 0.01
    dt: float = 0.002
    t_end: float = 1.0
    save_every: int = 5


class AllenCahnSolver:
    def __init__(self, n: int = 128, eps: float = 0.01, dt: float = 0.002):
        self.n = n
        self.eps = float(eps)
        self.dt = float(dt)
        self.k = wavenumbers_1d(n)
        self.k2 = self.k**2
        self.mask = dealias_mask_1d(n)

    def _reaction(self, u: np.ndarray) -> np.ndarray:
        return u - u**3

    def _diffuse_half(self, u: np.ndarray, dt: float) -> np.ndarray:
        uh = np.fft.fft(u, axis=-1) * np.exp(-self.eps * self.k2 * dt)
        return np.fft.ifft(uh, axis=-1).real

    def step(self, u: np.ndarray, dt: float | None = None) -> np.ndarray:
        dt = self.dt if dt is None else float(dt)
        u = self._diffuse_half(u, 0.5 * dt)
        k1 = self._reaction(u)
        k2 = self._reaction(u + 0.5 * dt * k1)
        k3 = self._reaction(u + 0.5 * dt * k2)
        k4 = self._reaction(u + dt * k3)
        u = u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return self._diffuse_half(u, 0.5 * dt)

    def temporal_residual(self, u_prev: np.ndarray, u_next: np.ndarray, dt: float,
                          eps: np.ndarray | float | None = None) -> np.ndarray:
        eps = self.eps if eps is None else eps
        uh = np.fft.fft(u_prev, axis=-1)
        lap = np.fft.ifft(-self.k2 * uh, axis=-1).real
        return (u_next - u_prev) / dt - eps * lap - (u_prev - u_prev**3)

    def free_energy(self, u: np.ndarray) -> float:
        """Ginzburg-Landau free energy; must be non-increasing in time."""
        ux = np.fft.ifft(1j * self.k * np.fft.fft(u, axis=-1), axis=-1).real
        dx = 2.0 * np.pi / self.n
        return float(np.mean(0.5 * self.eps * ux**2 + 0.25 * (u**2 - 1.0) ** 2) * dx)

    def solve(self, u0: np.ndarray, t_end: float = 1.0, save_every: int = 5):
        n_steps = int(round(t_end / self.dt))
        snaps = [u0.copy()]
        u = u0.copy()
        for s in range(1, n_steps + 1):
            u = self.step(u)
            if s % save_every == 0:
                snaps.append(u.copy())
        return np.stack(snaps, axis=0), self.dt * save_every
