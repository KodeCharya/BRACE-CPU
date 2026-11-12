"""Numerical correctness tests for the reference PDE solvers."""
from __future__ import annotations

import numpy as np
import pytest

from src.solvers.advection_diffusion import AdvectionDiffusionSolver
from src.solvers.allen_cahn import AllenCahnSolver
from src.solvers.burgers import BurgersSolver
from src.solvers.spectral import grad_1d, grf_1d, grf_2d


def test_grf_deterministic_and_zero_mean():
    a = grf_1d(64, np.random.default_rng(0), length_scale=0.4)
    b = grf_1d(64, np.random.default_rng(0), length_scale=0.4)
    assert np.allclose(a, b)
    assert abs(a.mean()) < 1e-10
    assert 0.5 < a.std() < 1.5


def test_spectral_gradient_sine():
    x = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    u = np.sin(3 * x)
    du = grad_1d(u, 2 * np.pi / 128)
    assert np.max(np.abs(du - 3 * np.cos(3 * x))) < 1e-9


def test_advection_diffusion_matches_analytic_mode():
    # Single Fourier mode decays and shifts exactly under the linear PDE.
    n, nu, cx = 64, 0.05, 1.0
    s = AdvectionDiffusionSolver(n=n, nu=nu, cx=cx, cy=0.0, dt=0.01)
    x = np.linspace(0, 2 * np.pi, n, endpoint=False)
    u0 = np.sin(x)[:, None].repeat(n, axis=1)
    out, dt_save = s.solve(u0, t_end=0.5, save_every=1)
    t = 0.5
    exact = np.exp(-nu * t) * np.sin(x - cx * t)[:, None].repeat(n, axis=1)
    err = np.linalg.norm(out[-1] - exact) / np.linalg.norm(exact)
    assert err < 1e-10, err
    assert np.isclose(dt_save, 0.01)


def test_advection_diffusion_residual_small():
    n = 32
    rng = np.random.default_rng(1)
    s = AdvectionDiffusionSolver(n=n, nu=0.02, cx=1.0, cy=0.5, dt=0.005)
    u0 = grf_2d(n, rng)
    out, dt = s.solve(u0, t_end=0.2, save_every=1)
    r = s.temporal_residual(out[0], out[1], dt)
    scale = np.abs(out[1] - out[0]).mean() / dt + 1e-8
    assert np.abs(r).mean() / scale < 2e-2


def test_advection_diffusion_batched_shapes():
    """The feature pipeline calls both methods with a batch of trajectories."""
    n, batch = 32, 4
    rng = np.random.default_rng(5)
    s = AdvectionDiffusionSolver(n=n, nu=0.02, cx=1.0, cy=0.5, dt=0.005)
    u = np.stack([grf_2d(n, rng) for _ in range(batch)])
    out = s.step(u)
    assert out.shape == u.shape
    r = s.temporal_residual(u, out, 0.005)
    assert r.shape == u.shape
    assert np.all(np.isfinite(r))


def test_burgers_energy_decays_and_residual_small():
    rng = np.random.default_rng(0)
    s = BurgersSolver(n=64, nu=0.02, dt=0.001)
    u0 = grf_1d(64, rng, length_scale=0.3, amplitude=0.5)
    out, dt = s.solve(u0, t_end=0.2, save_every=5)
    energy = np.mean(out**2, axis=1)
    assert np.all(np.diff(energy) < 1e-10)
    r = s.temporal_residual(out[0], out[1], dt)
    scale = np.abs(out[1] - out[0]).mean() / dt + 1e-8
    assert np.abs(r).mean() / scale < 2e-2


def test_burgers_temporal_convergence():
    rng = np.random.default_rng(3)
    u0 = grf_1d(64, rng, length_scale=0.5, amplitude=0.5)
    s1 = BurgersSolver(n=64, nu=0.02, dt=0.002)
    s2 = BurgersSolver(n=64, nu=0.02, dt=0.001)
    a, _ = s1.solve(u0, t_end=0.2, save_every=10)  # saves every 0.02
    b, _ = s2.solve(u0, t_end=0.2, save_every=20)
    # snapshots at same physical times
    assert a.shape == b.shape
    diff = np.linalg.norm((a - b)[-1]) / max(np.linalg.norm(b[-1]), 1e-12)
    assert diff < 0.02, diff


def test_allen_cahn_free_energy_decreases():
    rng = np.random.default_rng(2)
    s = AllenCahnSolver(n=64, eps=0.01, dt=0.002)
    u0 = 0.3 * grf_1d(64, rng, length_scale=0.3)
    out, dt = s.solve(u0, t_end=0.2, save_every=5)
    fe = np.array([s.free_energy(u) for u in out])
    # Strang splitting can fluctuate at O(dt^2); require no growth beyond noise.
    assert fe[-1] < fe[0]
    assert fe.max() - fe[0] < 1e-6


def test_allen_cahn_residual_small():
    rng = np.random.default_rng(4)
    s = AllenCahnSolver(n=64, eps=0.01, dt=0.001)
    u0 = 0.3 * grf_1d(64, rng)
    out, dt = s.solve(u0, t_end=0.1, save_every=5)
    r = s.temporal_residual(out[0], out[1], dt)
    scale = np.abs(out[1] - out[0]).mean() / dt + 1e-8
    assert np.abs(r).mean() / scale < 5e-3


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
