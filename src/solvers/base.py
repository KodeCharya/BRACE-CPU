"""Common base class and utilities for PDE reference solvers."""

from __future__ import annotations

import numpy as np


def periodic_gradient(u: np.ndarray, dx: float) -> np.ndarray:
    """Second-order central first derivative with periodic boundaries.

    (du/dx)[i] = (u[i+1] - u[i-1]) / (2 dx)
    """
    return (np.roll(u, -1) - np.roll(u, 1)) / (2.0 * dx)


def periodic_laplacian(u: np.ndarray, dx: float) -> np.ndarray:
    """Second-order central second derivative with periodic boundaries.

    (d2u/dx2)[i] = (u[i+1] - 2 u[i] + u[i-1]) / dx^2
    """
    return (np.roll(u, -1) - 2.0 * u + np.roll(u, 1)) / (dx ** 2)


def relative_l2_error(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    """Relative L2 error ||a-b||_2 / (||b||_2 + eps)."""
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + eps))


def max_abs_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


class PDESolver:
    """Base class for 1D periodic reference solvers.

    Subclasses implement ``rhs(u, t)`` returning du/dt. ``advance`` integrates
    from a given state using an explicit Runge-Kutta 4 scheme with a
    user-supplied (or auto-selected stable) time step.
    """

    def __init__(self, nx: int, L: float = 1.0):
        self.nx = int(nx)
        self.L = float(L)
        self.dx = self.L / self.nx
        self.x = np.linspace(0.0, self.L, self.nx, endpoint=False)

    def rhs(self, u: np.ndarray, t: float) -> np.ndarray:
        raise NotImplementedError

    def stable_dt(self, u: np.ndarray, safety: float = 0.5) -> float:
        """Conservative explicit-stability time-step estimate.

        Combines an advection CFL bound (dt <= dx / max|u|) and a diffusion
        bound (dt <= dx^2 / (2 * nu_eff)). Returns safety * min(bounds).
        """
        raise NotImplementedError

    def rk4_step(self, u: np.ndarray, t: float, dt: float) -> np.ndarray:
        k1 = self.rhs(u, t)
        k2 = self.rhs(u + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = self.rhs(u + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = self.rhs(u + dt * k3, t + dt)
        return u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def advance(self, u0: np.ndarray, t0: float, dt: float, n_steps: int) -> np.ndarray:
        """Advance state u0 forward n_steps of size dt via RK4.

        Returns a copy; u0 is not modified.
        """
        u = np.array(u0, dtype=np.float64, copy=True)
        t = t0
        for _ in range(n_steps):
            u = self.rk4_step(u, t, dt)
            t += dt
        return u

    def solve_trajectory(self, u0: np.ndarray, t0: float, t_end: float, dt: float):
        """Full trajectory: returns times array and state array (n_steps+1, nx)."""
        n_steps = int(round((t_end - t0) / dt))
        times = t0 + dt * np.arange(n_steps + 1)
        states = np.empty((n_steps + 1, self.nx), dtype=np.float64)
        states[0] = u0
        u = np.array(u0, dtype=np.float64, copy=True)
        t = t0
        for i in range(1, n_steps + 1):
            u = self.rk4_step(u, t, dt)
            t += dt
            states[i] = u
        return times, states
