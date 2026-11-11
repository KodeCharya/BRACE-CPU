"""1D viscous Burgers equation on a periodic domain.

    u_t + u u_x = nu u_xx,   x in [0, 2*pi),  t in [0, T]

Space: Fourier pseudo-spectral with 2/3 dealiasing.
Time:  Strang splitting (exact integrating factor for diffusion) with an
       explicit classical RK4 step for the nonlinear advection term.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .spectral import dealias_mask_1d, grad_1d, wavenumbers_1d


@dataclass
class BurgersConfig:
    n: int = 128
    nu: float = 0.01
    dt: float = 0.002
    t_end: float = 2.0
    save_every: int = 10


class BurgersSolver:
    def __init__(self, n: int = 128, nu: float = 0.01, dt: float = 0.002):
        if n % 2 != 0:
            raise ValueError("n must be even for the real FFT grid")
        self.n = n
        self.nu = float(nu)
        self.dt = float(dt)
        self.k = wavenumbers_1d(n)
        self.k2 = self.k**2
        self.mask = dealias_mask_1d(n)
        self.dx = 2.0 * np.pi / n

    def _nonlinear(self, u: np.ndarray) -> np.ndarray:
        uh = np.fft.fft(u, axis=-1) * self.mask
        ux = np.fft.ifft(1j * self.k * uh, axis=-1).real
        prod = u * ux
        ph = np.fft.fft(prod, axis=-1) * self.mask
        return -np.fft.ifft(ph, axis=-1).real

    def _diffuse_half(self, u: np.ndarray, dt: float) -> np.ndarray:
        uh = np.fft.fft(u, axis=-1)
        uh = uh * np.exp(-self.nu * self.k2 * dt)
        return np.fft.ifft(uh, axis=-1).real

    def step(self, u: np.ndarray, dt: float | None = None) -> np.ndarray:
        dt = self.dt if dt is None else float(dt)
        u = self._diffuse_half(u, 0.5 * dt)
        k1 = self._nonlinear(u)
        k2 = self._nonlinear(u + 0.5 * dt * k1)
        k3 = self._nonlinear(u + 0.5 * dt * k2)
        k4 = self._nonlinear(u + dt * k3)
        u = u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return self._diffuse_half(u, 0.5 * dt)

    def residual(self, u: np.ndarray) -> np.ndarray:
        """PDE residual u_t + u u_x - nu u_xx estimated with a finite difference
        in time is not available from a single snapshot; callers pass two
        consecutive states to :meth:`temporal_residual`."""
        raise NotImplementedError("use temporal_residual(u_prev, u_next, dt)")

    def temporal_residual(self, u_prev: np.ndarray, u_next: np.ndarray, dt: float,
                          nu: np.ndarray | float | None = None) -> np.ndarray:
        """Backward-Euler style residual of the PDE over one saved interval.

        ``nu`` may be a per-trajectory array broadcastable against ``u_prev``.
        """
        nu = self.nu if nu is None else nu
        ux = grad_1d(u_prev, self.dx)
        uxx = np.fft.ifft(-(self.k**2) * np.fft.fft(u_prev, axis=-1), axis=-1).real
        return (u_next - u_prev) / dt + u_prev * ux - nu * uxx

    def solve(self, u0: np.ndarray, t_end: float = 2.0, save_every: int = 10):
        """Roll forward and return snapshots of shape [nt, n] and saved dt."""
        n_steps = int(round(t_end / self.dt))
        snaps = [u0.copy()]
        u = u0.copy()
        for s in range(1, n_steps + 1):
            u = self.step(u)
            if s % save_every == 0:
                snaps.append(u.copy())
        arr = np.stack(snaps, axis=0)
        return arr, self.dt * save_every
