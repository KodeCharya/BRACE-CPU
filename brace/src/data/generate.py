"""Reproducible dataset generation for the BRACE benchmarks.

Each benchmark produces trajectories of shape [n_traj, n_time, *spatial].
Partitions (train/val/test/ood_*) are generated from disjoint SeedSequence
spawn keys and are never mixed. Metadata (configuration, seeds, SHA-256) is
written next to every array.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass

import numpy as np

from ..solvers.advection_diffusion import AdvectionDiffusionSolver
from ..solvers.allen_cahn import AllenCahnSolver
from ..solvers.burgers import BurgersSolver
from ..solvers.spectral import grf_1d, grf_2d

BENCHMARKS = ("burgers1d", "advection2d", "allencahn1d")

SPLITS = ("train", "val", "test", "ood_param", "ood_ic")

# Fixed partition sizes. Kept small enough for CPU training while remaining
# statistically useful (paired tests use trajectory-level means).
SPLIT_SIZES = {
    "train": 320,
    "val": 80,
    "test": 80,
    "ood_param": 60,
    "ood_ic": 60,
}


@dataclass
class TrajectorySpec:
    benchmark: str
    split: str
    index: int
    seed: int
    nu: float
    length_scale: float
    amplitude: float
    cx: float = 1.0
    cy: float = 0.5
    eps: float = 0.01


def _draw_spec(benchmark: str, split: str, index: int, base_seed: int) -> TrajectorySpec:
    seed = int(np.random.SeedSequence([base_seed, SPLITS.index(split), index]).generate_state(1)[0])
    rng = np.random.default_rng(seed)
    if benchmark == "burgers1d":
        if split == "ood_param":
            nu = float(np.exp(rng.uniform(np.log(0.0015), np.log(0.0035))))
            ls = rng.uniform(0.2, 0.5)
            amp = rng.uniform(0.6, 1.0)
        elif split == "ood_ic":
            nu = float(np.exp(rng.uniform(np.log(0.005), np.log(0.02))))
            ls = rng.uniform(0.7, 1.0)
            amp = rng.uniform(1.25, 1.6)
        else:
            nu = float(np.exp(rng.uniform(np.log(0.005), np.log(0.02))))
            ls = rng.uniform(0.2, 0.5)
            amp = rng.uniform(0.6, 1.0)
        return TrajectorySpec(benchmark, split, index, seed, nu, ls, amp)
    if benchmark == "allencahn1d":
        if split == "ood_param":
            eps = float(rng.uniform(0.004, 0.0065))
            ls = rng.uniform(0.2, 0.4)
            amp = rng.uniform(0.2, 0.45)
        elif split == "ood_ic":
            eps = 0.01
            ls = rng.uniform(0.6, 0.9)
            amp = rng.uniform(0.65, 0.85)
        else:
            eps = float(rng.uniform(0.008, 0.014))
            ls = rng.uniform(0.2, 0.4)
            amp = rng.uniform(0.2, 0.45)
        return TrajectorySpec(benchmark, split, index, seed, 0.0, ls, amp, eps=eps)
    if benchmark == "advection2d":
        if split == "ood_param":
            nu = float(rng.uniform(0.002, 0.005))
            cx = float(rng.uniform(1.8, 2.4))
            cy = float(rng.uniform(0.9, 1.2))
        elif split == "ood_ic":
            nu = float(rng.uniform(0.005, 0.02))
            cx = float(rng.uniform(0.5, 1.5))
            cy = float(rng.uniform(0.3, 0.8))
        else:
            nu = float(rng.uniform(0.005, 0.02))
            cx = float(rng.uniform(0.5, 1.5))
            cy = float(rng.uniform(0.3, 0.8))
        ls = rng.uniform(0.25, 0.45) if split != "ood_ic" else rng.uniform(0.06, 0.12)
        amp = rng.uniform(0.6, 1.0) if split != "ood_ic" else rng.uniform(0.15, 0.3)
        return TrajectorySpec(benchmark, split, index, seed, nu, ls, amp, cx=cx, cy=cy)
    raise ValueError(benchmark)


def _simulate(spec: TrajectorySpec) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(spec.seed + 7919)
    if spec.benchmark == "burgers1d":
        solver = BurgersSolver(n=128, nu=spec.nu, dt=0.002)
        u0 = grf_1d(128, rng, length_scale=spec.length_scale, amplitude=spec.amplitude)
        snaps, dt_save = solver.solve(u0, t_end=2.0, save_every=10)
    elif spec.benchmark == "allencahn1d":
        solver = AllenCahnSolver(n=128, eps=spec.eps, dt=0.002)
        u0 = grf_1d(128, rng, length_scale=spec.length_scale, amplitude=spec.amplitude)
        snaps, dt_save = solver.solve(u0, t_end=1.0, save_every=5)
    elif spec.benchmark == "advection2d":
        solver = AdvectionDiffusionSolver(n=64, nu=spec.nu, cx=spec.cx, cy=spec.cy, dt=0.01)
        u0 = grf_2d(64, rng, length_scale=spec.length_scale, amplitude=spec.amplitude)
        snaps, dt_save = solver.solve(u0, t_end=2.0, save_every=2)
    else:
        raise ValueError(spec.benchmark)
    return snaps.astype(np.float32), float(dt_save)


def generate_split(benchmark: str, split: str, base_seed: int = 20261111,
                   out_dir: str = "data/processed", n_traj: int | None = None,
                   verbose: bool = True) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    n = SPLIT_SIZES[split] if n_traj is None else n_traj
    trajs = []
    specs = []
    t0 = time.time()
    for i in range(n):
        spec = _draw_spec(benchmark, split, i, base_seed)
        arr, dt_save = _simulate(spec)
        trajs.append(arr)
        specs.append(asdict(spec))
        if verbose and (i + 1) % 40 == 0:
            print(f"  {benchmark}/{split}: {i + 1}/{n} ({time.time() - t0:.1f}s)", flush=True)
    data = np.stack(trajs, axis=0)
    path = os.path.join(out_dir, f"{benchmark}_{split}.npz")
    np.savez_compressed(
        path,
        u=data,
        dt_save=np.float32(dt_save),
        nu=np.array([s["nu"] for s in specs], dtype=np.float32),
        cx=np.array([s["cx"] for s in specs], dtype=np.float32),
        cy=np.array([s["cy"] for s in specs], dtype=np.float32),
        eps=np.array([s["eps"] for s in specs], dtype=np.float32),
        length_scale=np.array([s["length_scale"] for s in specs], dtype=np.float32),
        amplitude=np.array([s["amplitude"] for s in specs], dtype=np.float32),
    )
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    meta = {
        "benchmark": benchmark,
        "split": split,
        "n_traj": int(n),
        "shape": list(data.shape),
        "dtype": str(data.dtype),
        "dt_save": float(dt_save),
        "base_seed": base_seed,
        "sha256": sha,
        "generation_seconds": round(time.time() - t0, 2),
        "specs": specs,
        "config": _benchmark_config(benchmark),
    }
    meta_path = os.path.join(out_dir, "..", "metadata", f"{benchmark}_{split}.json")
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    if verbose:
        print(f"saved {path} shape={data.shape} sha256={sha[:16]}...", flush=True)
    return meta


def _benchmark_config(benchmark: str) -> dict:
    if benchmark == "burgers1d":
        return {
            "equation": "u_t + u u_x = nu u_xx",
            "domain": "[0, 2*pi) periodic",
            "n": 128, "dt_internal": 0.002, "dt_save": 0.02, "t_end": 2.0,
            "nu_train_range": [0.005, 0.02],
            "ic": "periodic GRF, squared-exponential spectrum, length_scale U(0.2,0.5), amplitude U(0.6,1.0)",
            "solver": "Fourier pseudo-spectral + 2/3 dealias, Strang splitting, RK4 advection",
        }
    if benchmark == "allencahn1d":
        return {
            "equation": "u_t = eps u_xx + u - u^3",
            "domain": "[0, 2*pi) periodic",
            "n": 128, "dt_internal": 0.002, "dt_save": 0.01, "t_end": 1.0,
            "eps_train_range": [0.008, 0.014],
            "ic": "periodic GRF amplitude U(0.2,0.45)",
            "solver": "Fourier pseudo-spectral, Strang splitting, RK4 reaction",
        }
    return {
        "equation": "u_t + cx u_x + cy u_y = nu (u_xx + u_yy)",
        "domain": "[0, 2*pi)^2 periodic",
        "n": 64, "dt_internal": 0.01, "dt_save": 0.02, "t_end": 2.0,
        "nu_train_range": [0.005, 0.02], "velocity_train_range": [0.5, 1.5, 0.3, 0.8],
        "ic": "periodic 2D GRF, length_scale U(0.25,0.45), amplitude U(0.6,1.0)",
        "solver": "exact Fourier symbol integrator (linear PDE)",
    }
