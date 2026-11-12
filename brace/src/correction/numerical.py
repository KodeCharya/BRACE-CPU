"""Numerical correction engine.

Correction integrates the governing PDE from a (possibly surrogate) state
using the high-accuracy reference solver.

Cost is measured in *reference substeps*. One saved interval costs
``n_sub`` substeps; a full numerical simulation of the horizon costs
``T * n_sub``. The budget fraction B is therefore

    B = solver_substeps_used / (T * n_sub)

which is directly comparable to the cost of just running the reference
solver for the whole horizon (B = 1).
"""
from __future__ import annotations

import numpy as np

from ..solvers.advection_diffusion import AdvectionDiffusionSolver
from ..solvers.allen_cahn import AllenCahnSolver
from ..solvers.burgers import BurgersSolver


def default_n(benchmark: str) -> int:
    return {"burgers1d": 128, "allencahn1d": 128, "advection2d": 64}[benchmark]


def make_solver(benchmark: str, params: dict, dt_save: float, n: int | None = None):
    n = default_n(benchmark) if n is None else int(n)
    if benchmark == "burgers1d":
        return BurgersSolver(n=n, nu=float(params["nu"]), dt=dt_save), dt_save
    if benchmark == "allencahn1d":
        return AllenCahnSolver(n=n, eps=float(params["eps"]), dt=dt_save), dt_save
    if benchmark == "advection2d":
        return AdvectionDiffusionSolver(n=n, nu=float(params["nu"]), cx=float(params["cx"]),
                                        cy=float(params["cy"]), dt=dt_save), dt_save
    raise ValueError(benchmark)


def n_substeps(benchmark: str) -> int:
    """Reference substeps per saved interval used during data generation."""
    return {"burgers1d": 10, "allencahn1d": 5, "advection2d": 2}[benchmark]


class Corrector:
    """Applies full-domain numerical integration to selected trajectories."""

    def __init__(self, benchmark: str, params: dict, dt_save: float):
        self.benchmark = benchmark
        self.dt_save = dt_save
        self.n_sub = n_substeps(benchmark)
        self.solvers = {}
        self.params = params

    def _solver(self, i: int, n: int | None = None):
        key = (i, n)
        if key not in self.solvers:
            p = {k: float(v[i]) for k, v in self.params.items() if k in ("nu", "eps", "cx", "cy")}
            # Use the same internal substep as dataset generation so that a
            # "correction" has reference-solver accuracy.
            self.solvers[key] = make_solver(self.benchmark, p,
                                            self.dt_save / self.n_sub, n=n)[0]
        return self.solvers[key]

    def advance(self, u: np.ndarray, i: int, n_intervals: int = 1) -> np.ndarray:
        """Advance state u for n_intervals saved intervals with the reference solver."""
        s = self._solver(i, u.shape[-1])
        out = np.array(u, copy=True)
        for _ in range(n_intervals * self.n_sub):
            out = s.step(out)
        return out

    def cost_units(self, n_intervals: int = 1) -> int:
        return n_intervals * self.n_sub
