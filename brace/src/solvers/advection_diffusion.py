"""2D linear advection-diffusion equation on a periodic domain.

    u_t + c . grad(u) = nu * Laplacian(u),   x in [0, 2*pi)^2

The equation is linear with constant coefficients, so the Fourier symbol
gives an exact time integrator (no splitting error):

    u_hat(t+dt) = u_hat(t) * exp((-i k.c - nu |k|^2) dt)

Space: Fourier pseudo-spectral with 2/3 dealiasing of the initial field.
This benchmark isolates distribution shift driven by the transport
velocity and the diffusion coefficient.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .spectral import dealias_mask_2d, wavenumbers_2d


@dataclass
class AdvectionDiffusionConfig:
    n: int = 64
    nu: float = 0.01
    cx: float = 1.0
    cy: float = 0.5
    dt: float = 0.01
    t_end: float = 2.0
    save_every: int = 2


class AdvectionDiffusionSolver:
    def __init__(self, n: int = 64, nu: float = 0.01, cx: float = 1.0, cy: float = 0.5,
                 dt: float = 0.01):
        self.n = n
        self.nu = float(nu)
        self.cx = float(cx)
        self.cy = float(cy)
        self.dt = float(dt)
        kx, ky = wavenumbers_2d(n)
        self.kx, self.ky = kx, ky
        self.k2 = kx**2 + ky**2
        self.mask = dealias_mask_2d(n)

    def step(self, u: np.ndarray, dt: float | None = None) -> np.ndarray:
        dt = self.dt if dt is None else float(dt)
        uh = np.fft.rfft2(u, axes=(-2, -1))
        symbol = np.exp((-1j * (self.kx * self.cx + self.ky * self.cy) - self.nu * self.k2) * dt)
        return np.fft.irfft2(uh * symbol, s=u.shape[-2:], axes=(-2, -1)).real

    def temporal_residual(self, u_prev: np.ndarray, u_next: np.ndarray, dt: float,
                          nu: np.ndarray | float | None = None,
                          cx: np.ndarray | float | None = None,
                          cy: np.ndarray | float | None = None) -> np.ndarray:
        nu = self.nu if nu is None else nu
        cx = self.cx if cx is None else cx
        cy = self.cy if cy is None else cy
        uh = np.fft.rfft2(u_prev, axes=(-2, -1))
        ux = np.fft.irfft2(1j * self.kx * uh, s=u_prev.shape[-2:], axes=(-2, -1)).real
        uy = np.fft.irfft2(1j * self.ky * uh, s=u_prev.shape[-2:], axes=(-2, -1)).real
        lap = np.fft.irfft2(-self.k2 * uh, s=u_prev.shape[-2:], axes=(-2, -1)).real
        return (u_next - u_prev) / dt + cx * ux + cy * uy - nu * lap

    def solve(self, u0: np.ndarray, t_end: float = 2.0, save_every: int = 2):
        n_steps = int(round(t_end / self.dt))
        snaps = [u0.copy()]
        u = u0.copy()
        for s in range(1, n_steps + 1):
            u = self.step(u)
            if s % save_every == 0:
                snaps.append(u.copy())
        return np.stack(snaps, axis=0), self.dt * save_every
