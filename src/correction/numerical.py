"""Numerical correction strategies.

When the controller invokes the numerical solver, it advances from the
currently available state (the surrogate's current state). No future ground
truth is accessed. The primary strategy is one-step numerical advancement
from the current state (full-domain).
"""

from __future__ import annotations

import numpy as np


def correct_one_step(solver, u: np.ndarray, t: float, dt: float) -> np.ndarray:
    """Advance one numerical step of size dt from state u (full domain)."""
    return solver.advance(u, t, dt, 1)


def correct_multi_step(solver, u: np.ndarray, t: float, dt: float, n_substeps: int) -> np.ndarray:
    """Advance n_substeps numerical steps of size dt from state u."""
    return solver.advance(u, t, dt, n_substeps)


def full_fallback(solver, u: np.ndarray, t: float, t_end: float, dt: float):
    """Numerically advance from u at time t to t_end (full numerical fallback)."""
    n_steps = int(round((t_end - t) / dt))
    u_out = solver.advance(u, t, dt, n_steps)
    return u_out, t + n_steps * dt
