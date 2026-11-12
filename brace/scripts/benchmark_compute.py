"""Measure inference and solver latency on the target CPU machine.

Reports per-model latency, reference-solver cost per saved interval, the
planning overhead of the BRACE controller, and peak memory. Run on an idle
machine: wall-clock numbers under CPU contention are not meaningful.
"""
from __future__ import annotations

import json
import os
import resource
import time

import numpy as np
import torch

from src.models.fno import build_model
from src.solvers.advection_diffusion import AdvectionDiffusionSolver
from src.solvers.allen_cahn import AllenCahnSolver
from src.solvers.burgers import BurgersSolver
from src.solvers.spectral import grf_1d, grf_2d

RESULTS = {}


def timeit(fn, n: int = 20, warmup: int = 5) -> float:
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n


def model_latency(name: str, model, x: torch.Tensor, n: int = 20) -> dict:
    model.eval()
    with torch.no_grad():
        dt = timeit(lambda: model(x), n=n)
    return {"params": sum(p.numel() for p in model.parameters()),
            "batch": int(x.shape[0]),
            "latency_ms_per_batch": dt * 1e3,
            "latency_ms_per_sample": dt * 1e3 / x.shape[0]}


def main() -> None:
    torch.set_num_threads(4)
    out = {"device": "cpu", "torch_threads": torch.get_num_threads(),
           "torch_version": torch.__version__, "cuda_available": torch.cuda.is_available(),
           "cpu_count": os.cpu_count()}
    models = {}
    models["fno1d_w32_m32_l4"] = model_latency(
        "fno1d", build_model("fno1d", width=32, modes=32, layers=4),
        torch.randn(40, 128))
    models["fno1d_w32_m24_l4"] = model_latency(
        "fno1d", build_model("fno1d", width=32, modes=24, layers=4),
        torch.randn(40, 128))
    models["fno2d_w16_m8_l3_32x32"] = model_latency(
        "fno2d", build_model("fno2d", width=16, modes1=8, modes2=8, layers=3),
        torch.randn(24, 32, 32), n=10)
    models["deeponet1d_p128"] = model_latency(
        "deeponet1d", build_model("deeponet1d", n_points=128),
        torch.randn(40, 128))
    out["models"] = models

    # Reference solver: cost per saved interval (n_sub internal substeps).
    solvers = {}
    u1 = grf_1d(128, np.random.default_rng(0))
    s = BurgersSolver(n=128, nu=0.01, dt=0.002)
    solvers["burgers1d"] = {"substep_ms": timeit(lambda: s.step(u1), n=50) * 1e3,
                            "substeps_per_interval": 10,
                            "interval_ms": timeit(lambda: s.step(u1), n=50) * 1e3 * 10}
    s = AllenCahnSolver(n=128, eps=0.01, dt=0.002)
    solvers["allencahn1d"] = {"substep_ms": timeit(lambda: s.step(u1), n=50) * 1e3,
                              "substeps_per_interval": 5,
                              "interval_ms": timeit(lambda: s.step(u1), n=50) * 1e3 * 5}
    u2 = grf_2d(32, np.random.default_rng(0))
    s = AdvectionDiffusionSolver(n=32, nu=0.01, cx=1.0, cy=0.5, dt=0.01)
    solvers["advection2d"] = {"substep_ms": timeit(lambda: s.step(u2), n=50) * 1e3,
                              "substeps_per_interval": 2,
                              "interval_ms": timeit(lambda: s.step(u2), n=50) * 1e3 * 2}
    out["reference_solver"] = solvers

    # Planning overhead: 3-member ensemble rollout over 40 trajectories x 100 steps.
    from src.controller.rollout import neural_pass
    from src.uncertainty.pipeline import residual_solver
    ens = [build_model("fno1d", width=32, modes=32, layers=4).eval() for _ in range(3)]
    z = np.load("data/processed/burgers1d_test.npz")
    u = z["u"][:40].astype(np.float32)
    nu = {"nu": z["nu"][:40]}
    t0 = time.perf_counter()
    feats, errs = neural_pass(ens, u, residual_solver("burgers1d", float(z["dt_save"])),
                              nu, float(z["dt_save"]))
    plan_s = time.perf_counter() - t0
    out["planning"] = {"trajectories": int(u.shape[0]), "steps": int(u.shape[1] - 1),
                       "ensemble": 3, "seconds": plan_s,
                       "ms_per_trajectory_step": plan_s * 1e3 / (u.shape[0] * (u.shape[1] - 1))}

    out["memory"] = {
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "system_total_mib": os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024 ** 2,
    }
    os.makedirs("results/processed", exist_ok=True)
    with open("results/processed/compute_benchmark.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
