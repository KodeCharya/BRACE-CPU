"""1D linear advection-diffusion reference solver: u_t + a u_x = D u_xx.

Spatial: 2nd-order central differences (periodic).
Time: explicit RK4 with user-supplied or auto stability-limited dt.
Precision: float64.
"""

from __future__ import annotations

import numpy as np

from .base import PDESolver, periodic_gradient, periodic_laplacian


class AdvectionDiffusionSolver(PDESolver):
    def __init__(self, nx: int, L: float = 1.0, a: float = 1.0, D: float = 0.01):
        super().__init__(nx=nx, L=L)
        self.a = float(a)
        self.D = float(D)

    def rhs(self, u: np.ndarray, t: float) -> np.ndarray:
        adv = self.a * periodic_gradient(u, self.dx)      # a u_x
        diff = self.D * periodic_laplacian(u, self.dx)    # D u_xx
        return -adv + diff

    def stable_dt(self, u: np.ndarray, safety: float = 0.5) -> float:
        cfl_dt = self.dx / abs(self.a) if self.a != 0 else np.inf
        diff_dt = (self.dx ** 2) / (2.0 * self.D) if self.D > 0 else np.inf
        return safety * min(cfl_dt, diff_dt)
