"""1D viscous Burgers' equation reference solver: u_t + u u_x = nu u_xx.

Conservative form: u_t + (0.5 u^2)_x = nu u_xx.
Spatial: 2nd-order central differences (periodic).
Time: explicit RK4 with user-supplied or auto stability-limited dt.
Precision: float64.
"""

from __future__ import annotations

import numpy as np

from .base import PDESolver, periodic_gradient, periodic_laplacian, relative_l2_error, max_abs_error


class BurgersSolver(PDESolver):
    def __init__(self, nx: int, L: float = 1.0, nu: float = 0.01):
        super().__init__(nx=nx, L=L)
        self.nu = float(nu)

    def rhs(self, u: np.ndarray, t: float) -> np.ndarray:
        # Flux form: u_t + d(0.5 u^2)/dx = nu u_xx
        flux = 0.5 * u ** 2
        adv = periodic_gradient(flux, self.dx)        # d(0.5 u^2)/dx
        diff = self.nu * periodic_laplacian(u, self.dx)  # nu u_xx
        return -adv + diff

    def stable_dt(self, u: np.ndarray, safety: float = 0.5) -> float:
        max_abs_u = float(np.max(np.abs(u))) if u.size else 0.0
        cfl_dt = self.dx / max_abs_u if max_abs_u > 0 else np.inf
        diff_dt = (self.dx ** 2) / (2.0 * self.nu) if self.nu > 0 else np.inf
        return safety * min(cfl_dt, diff_dt)

    def solve(self, u0: np.ndarray, t0: float, t_end: float, dt: float | None = None):
        """Convenience: auto-select stable dt if not provided, then solve."""
        u0 = np.asarray(u0, dtype=np.float64)
        if dt is None:
            dt = self.stable_dt(u0)
        return self.solve_trajectory(u0, t0, t_end, dt)
