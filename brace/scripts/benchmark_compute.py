"""Measure inference/solver latency, throughput and memory (CPU-only here)."""
from __future__ import annotations

import json
import os
import resource
import time

import numpy as np
import torch

from src.models.fno import build_model
from src.solvers.burgers import BurgersSolver
from src.solvers.spectral import grf_1d


def timeit(fn, n=30, warmup=5):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n


def main() -> None:
    torch.set_num_threads(4)
    out = {"device": "cpu", "torch_threads": torch.get_num_threads(),
           "torch_version": torch.__version__, "cuda_available": torch.cuda.is_available()}
    results = {}
    for name, cfg, shape in [
        ("fno1d_w32_m32_l4", dict(width=32, modes=32, layers=4), (1, 128)),
        ("fno1d_w24_m12_l3", dict(width=24, modes=12, layers=3), (1, 128)),
        ("fno2d_w16_m8_l3", dict(width=16, modes1=8, modes2=8, layers=3), (1, 32, 32)),
    ]:
        model = build_model("fno2d" if "2d" in name else "fno1d", **cfg).eval()
        x = torch.randn(*shape)
        with torch.no_grad():
            dt = timeit(lambda: model(x), n=20)
        results[name] = {
            "params": sum(p.numel() for p in model.parameters()),
            "latency_ms_per_sample": dt * 1e3,
            "samples_per_second": 1.0 / dt,
        }
    # Batched ensemble latency (3 members, batch 40).
    models = [build_model("fno1d", width=32, modes=32, layers=4).eval() for _ in range(3)]
    xb = torch.randn(40, 128)
    with torch.no_grad():
        dt = timeit(lambda: [m(xb) for m in models], n=10)
    results["ensemble3_batch40"] = {"latency_ms_per_batch": dt * 1e3,
                                    "latency_ms_per_sample": dt * 1e3 / 40}

    # Reference solver step latency versus a full trajectory.
    s = BurgersSolver(n=128, nu=0.01, dt=0.002)
    u = grf_1d(128, np.random.default_rng(0))
    dt_step = timeit(lambda: s.step(u), n=50)
    u0 = grf_1d(128, np.random.default_rng(1))
    t0 = time.perf_counter()
    s.solve(u0, t_end=2.0, save_every=10)
    t_traj = time.perf_counter() - t0
    results["burgers_solver"] = {
        "single_substep_ms": dt_step * 1e3,
        "full_trajectory_seconds": t_traj,
        "substeps_per_trajectory": 1000,
    }
    results["memory"] = {
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "system_total_mib": os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**2,
    }
    out["results"] = results
    os.makedirs("results/processed", exist_ok=True)
    with open("results/processed/compute_benchmark.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
