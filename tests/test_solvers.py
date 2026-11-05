"""Automated tests for the numerical reference solvers.

Run from project root:
    PYTHONPATH=src .venv/bin/python -m pytest tests/ -v
"""

from __future__ import annotations

import numpy as np
import pytest

from solvers.base import (
    periodic_gradient,
    periodic_laplacian,
    relative_l2_error,
    max_abs_error,
)
from solvers.burgers import BurgersSolver
from solvers.advection_diffusion import AdvectionDiffusionSolver


# ---------------------------------------------------------------------------
# Manufactured exact solution for advection-diffusion:
#   u(x,t) = exp(-D k^2 t) sin(k (x - a t)),  k = 2 pi n / L
# satisfies u_t + a u_x = D u_xx exactly with periodic BCs.
# ---------------------------------------------------------------------------

def _exact_advection_diffusion(x, t, a, D, n_mode=1, L=1.0):
    k = 2.0 * np.pi * n_mode / L
    return np.exp(-D * k ** 2 * t) * np.sin(k * (x - a * t))


def test_periodic_gradient_and_laplacian_are_periodic():
    nx = 64
    L = 1.0
    dx = L / nx
    x = np.linspace(0.0, L, nx, endpoint=False)
    u = np.sin(2.0 * np.pi * x)
    gu = periodic_gradient(u, dx)
    lu = periodic_laplacian(u, dx)
    # roll invariance: shifting input shifts output by same amount
    shift = 7
    u_shifted = np.roll(u, shift)
    assert np.allclose(periodic_gradient(u_shifted, dx), np.roll(gu, shift), atol=1e-12)
    assert np.allclose(periodic_laplacian(u_shifted, dx), np.roll(lu, shift), atol=1e-12)
    # laplacian of sin(kx) = -(4/dx^2) sin^2(k dx/2) sin(kx)  (discrete eigenvalue)
    k = 2.0 * np.pi / L
    eig = -4.0 * np.sin(k * dx / 2.0) ** 2 / dx ** 2
    assert np.allclose(lu, eig * u, atol=1e-12)


def test_advection_diffusion_exact_solution_convergence():
    L = 1.0
    a = 1.0
    D = 0.01
    t_end = 0.2
    n_mode = 1
    errors = []
    for nx in [32, 64, 128]:
        solver = AdvectionDiffusionSolver(nx=nx, L=L, a=a, D=D)
        x = solver.x
        u0 = _exact_advection_diffusion(x, 0.0, a, D, n_mode, L)
        # stable dt
        dt = solver.stable_dt(u0, safety=0.4)
        n_steps = int(round(t_end / dt))
        _, states = solver.solve_trajectory(u0, 0.0, t_end, dt)
        u_end = states[-1]
        u_exact = _exact_advection_diffusion(x, t_end, a, D, n_mode, L)
        err = relative_l2_error(u_end, u_exact)
        errors.append((nx, dt, err))
    # Error should decrease with refinement (2nd order in space)
    assert errors[0][2] > errors[1][2] or np.isclose(errors[0][2], errors[1][2], rtol=0.5)
    assert errors[1][2] > errors[2][2] or np.isclose(errors[1][2], errors[2][2], rtol=0.5)
    # Final error should be small
    assert errors[-1][2] < 1e-2, errors


def test_advection_diffusion_conserves_mass_when_a_equals_zero_diffusion_only():
    # Pure diffusion with periodic BC conserves integral of u (mean constant).
    L = 1.0
    a = 0.0
    D = 0.01
    t_end = 0.1
    nx = 64
    solver = AdvectionDiffusionSolver(nx=nx, L=L, a=a, D=D)
    u0 = np.sin(2.0 * np.pi * solver.x) + 0.5  # nonzero mean
    dt = solver.stable_dt(u0, safety=0.4)
    n_steps = int(round(t_end / dt))
    _, states = solver.solve_trajectory(u0, 0.0, t_end, dt)
    assert np.isclose(np.mean(states[-1]), np.mean(u0), atol=1e-8)


def test_burgers_mass_conservation_periodic():
    # Burgers with periodic BC: d/dt int u dx = 0 (flux periodic -> mass conserved).
    L = 1.0
    nu = 0.01
    t_end = 0.05
    nx = 64
    solver = BurgersSolver(nx=nx, L=L, nu=nu)
    rng = np.random.default_rng(0)
    u0 = np.sin(2.0 * np.pi * solver.x) + 0.3 * rng.standard_normal(nx)
    u0 = u0 - u0.mean()  # zero mean to keep max|u| moderate
    dt = solver.stable_dt(u0, safety=0.4)
    n_steps = int(round(t_end / dt))
    _, states = solver.solve_trajectory(u0, 0.0, t_end, dt)
    assert np.isclose(np.mean(states[-1]), np.mean(u0), atol=1e-8)


def test_burgers_no_blowup_for_smooth_initial_data():
    L = 1.0
    nu = 0.05
    t_end = 0.5
    nx = 64
    solver = BurgersSolver(nx=nx, L=L, nu=nu)
    u0 = np.sin(2.0 * np.pi * solver.x)
    dt = solver.stable_dt(u0, safety=0.4)
    n_steps = int(round(t_end / dt))
    _, states = solver.solve_trajectory(u0, 0.0, t_end, dt)
    assert np.all(np.isfinite(states))
    assert float(np.max(np.abs(states))) < 10.0


def test_advance_does_not_modify_input():
    solver = AdvectionDiffusionSolver(nx=32, L=1.0, a=1.0, D=0.01)
    u0 = np.sin(2.0 * np.pi * solver.x)
    u0_copy = u0.copy()
    _ = solver.advance(u0, 0.0, 0.001, 5)
    assert np.allclose(u0, u0_copy)


def test_stable_dt_positive_and_shrinks_with_refinement():
    solver32 = AdvectionDiffusionSolver(nx=32, L=1.0, a=1.0, D=0.01)
    solver64 = AdvectionDiffusionSolver(nx=64, L=1.0, a=1.0, D=0.01)
    u32 = np.sin(2.0 * np.pi * solver32.x)
    u64 = np.sin(2.0 * np.pi * solver64.x)
    dt32 = solver32.stable_dt(u32)
    dt64 = solver64.stable_dt(u64)
    assert dt32 > 0 and dt64 > 0
    # Halving dx roughly quarters the diffusion-limited dt
    assert dt64 < dt32


def test_relative_l2_error_properties():
    a = np.array([1.0, 2.0, 3.0])
    assert relative_l2_error(a, a) == 0.0
    assert relative_l2_error(a, a + 1.0) > 0.0
    assert max_abs_error(a, a) == 0.0
