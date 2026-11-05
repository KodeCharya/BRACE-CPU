"""Dataset generation for BRACE-CPU.

Reproducible synthetic PDE trajectories simulated with the float64 reference
solvers. Each trajectory is saved as an .npz file with a sidecar JSON metadata
record; SHA-256 checksums are recorded in a dataset manifest.

Partitions: train / validation / calibration / test (no trajectory leakage).
An out-of-distribution (OOD) test partition uses parameter values / initial-condition
distributions not represented in training.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Tuple

import numpy as np

from solvers.burgers import BurgersSolver
from solvers.advection_diffusion import AdvectionDiffusionSolver


# ---------------------------------------------------------------------------
# Initial-condition generators
# ---------------------------------------------------------------------------

def _rng_for(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def random_fourier_initial_condition(
    nx: int,
    seed: int,
    L: float = 1.0,
    n_modes: int = 8,
    mean: float = 0.0,
    amplitude: float = 1.0,
    decay: bool = True,
) -> np.ndarray:
    """Random smooth periodic initial condition as a truncated Fourier series.

    Reproducible given ``seed``. Coefficients decay with mode number when
    ``decay=True``, producing smoother fields.
    """
    rng = _rng_for(seed)
    x = np.linspace(0.0, L, nx, endpoint=False)
    u = np.full(nx, float(mean), dtype=np.float64)
    for k in range(1, n_modes + 1):
        scale = (1.0 / k) if decay else 1.0
        a = rng.standard_normal() * amplitude * scale
        b = rng.standard_normal() * amplitude * scale
        u = u + a * np.sin(2.0 * np.pi * k * x / L) + b * np.cos(2.0 * np.pi * k * x / L)
    return u


def gaussian_random_field_initial_condition(
    nx: int, seed: int, L: float = 1.0, length_scale: float = 0.15, amplitude: float = 1.0
) -> np.ndarray:
    """Smooth periodic Gaussian random field via spectral filtering of white noise."""
    rng = _rng_for(seed)
    x = np.linspace(0.0, L, nx, endpoint=False)
    white = rng.standard_normal(nx)
    # Spectral filter: attenuate high modes
    freqs = np.fft.fftfreq(nx, d=L / nx)
    # Gaussian filter in Fourier space
    sigma = 1.0 / (2.0 * np.pi * length_scale)
    filt = np.exp(-0.5 * (2.0 * np.pi * freqs * sigma) ** 2)
    u_hat = np.fft.fft(white) * filt
    u = np.fft.ifft(u_hat).real
    u = u / (np.std(u) + 1e-12) * amplitude
    return u


# ---------------------------------------------------------------------------
# Trajectory generation
# ---------------------------------------------------------------------------

@dataclass
class TrajectoryMetadata:
    trajectory_id: str
    pde: str  # "burgers" or "advection_diffusion"
    partition: str  # train/validation/calibration/test/ood_test
    seed: int
    nx: int
    L: float
    dt: float
    t0: float
    t_end: float
    n_steps: int
    # PDE parameters
    nu: float | None = None
    a: float | None = None
    D: float | None = None
    # Initial condition
    ic_type: str = "fourier"
    ic_n_modes: int | None = None
    ic_mean: float = 0.0
    ic_amplitude: float = 1.0
    ic_length_scale: float | None = None
    # Solver
    solver: str = "rk4"
    spatial_scheme: str = "central_2nd"
    boundary: str = "periodic"
    precision: str = "float64"
    # checksum
    sha256: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _stable_dt_for(solver, u0: np.ndarray, safety: float = 0.35) -> float:
    return float(solver.stable_dt(u0, safety=safety))


def generate_burgers_trajectory(
    seed: int,
    nx: int = 64,
    L: float = 1.0,
    nu: float = 0.01,
    t_end: float = 0.5,
    dt: float | None = None,
    ic_type: str = "fourier",
    ic_n_modes: int = 8,
    ic_mean: float = 0.0,
    ic_amplitude: float = 1.0,
    safety: float = 0.35,
) -> Tuple[np.ndarray, np.ndarray, TrajectoryMetadata]:
    """Simulate one Burgers trajectory and return (times, states, metadata)."""
    rng = _rng_for(seed)
    if ic_type == "fourier":
        u0 = random_fourier_initial_condition(
            nx, seed, L=L, n_modes=ic_n_modes, mean=ic_mean, amplitude=ic_amplitude
        )
    elif ic_type == "grf":
        u0 = gaussian_random_field_initial_condition(
            nx, seed, L=L, length_scale=0.15, amplitude=ic_amplitude
        )
    else:
        raise ValueError(f"unknown ic_type {ic_type}")

    solver = BurgersSolver(nx=nx, L=L, nu=nu)
    if dt is None:
        dt = _stable_dt_for(solver, u0, safety=safety)
    n_steps = int(round(t_end / dt))
    times, states = solver.solve_trajectory(u0, 0.0, t_end, dt)

    meta = TrajectoryMetadata(
        trajectory_id=f"burgers_s{seed}_n{nx}_nu{nu}",
        pde="burgers",
        partition="",
        seed=seed,
        nx=nx,
        L=L,
        dt=dt,
        t0=0.0,
        t_end=t_end,
        n_steps=n_steps,
        nu=nu,
        ic_type=ic_type,
        ic_n_modes=ic_n_modes,
        ic_mean=ic_mean,
        ic_amplitude=ic_amplitude,
        solver="rk4",
        spatial_scheme="central_2nd",
        boundary="periodic",
        precision="float64",
    )
    return times, states, meta


def generate_advection_diffusion_trajectory(
    seed: int,
    nx: int = 64,
    L: float = 1.0,
    a: float = 1.0,
    D: float = 0.01,
    t_end: float = 0.5,
    dt: float | None = None,
    ic_type: str = "fourier",
    ic_n_modes: int = 8,
    ic_mean: float = 0.0,
    ic_amplitude: float = 1.0,
    safety: float = 0.35,
) -> Tuple[np.ndarray, np.ndarray, TrajectoryMetadata]:
    rng = _rng_for(seed)
    if ic_type == "fourier":
        u0 = random_fourier_initial_condition(
            nx, seed, L=L, n_modes=ic_n_modes, mean=ic_mean, amplitude=ic_amplitude
        )
    elif ic_type == "grf":
        u0 = gaussian_random_field_initial_condition(
            nx, seed, L=L, length_scale=0.15, amplitude=ic_amplitude
        )
    else:
        raise ValueError(f"unknown ic_type {ic_type}")

    solver = AdvectionDiffusionSolver(nx=nx, L=L, a=a, D=D)
    if dt is None:
        dt = _stable_dt_for(solver, u0, safety=safety)
    n_steps = int(round(t_end / dt))
    times, states = solver.solve_trajectory(u0, 0.0, t_end, dt)

    meta = TrajectoryMetadata(
        trajectory_id=f"advdiff_s{seed}_n{nx}_a{a}_D{D}",
        pde="advection_diffusion",
        partition="",
        seed=seed,
        nx=nx,
        L=L,
        dt=dt,
        t0=0.0,
        t_end=t_end,
        n_steps=n_steps,
        a=a,
        D=D,
        ic_type=ic_type,
        ic_n_modes=ic_n_modes,
        ic_mean=ic_mean,
        ic_amplitude=ic_amplitude,
        solver="rk4",
        spatial_scheme="central_2nd",
        boundary="periodic",
        precision="float64",
    )
    return times, states, meta


# ---------------------------------------------------------------------------
# Dataset assembly + persistence
# ---------------------------------------------------------------------------

def save_trajectory(
    times: np.ndarray,
    states: np.ndarray,
    meta: TrajectoryMetadata,
    directory: str,
) -> str:
    """Save trajectory + metadata; return the .npz path. Updates meta.sha256."""
    os.makedirs(directory, exist_ok=True)
    traj_id = meta.trajectory_id
    npz_path = os.path.join(directory, f"{traj_id}.npz")
    # Store states in float32 to halve storage; times float64.
    np.savez_compressed(
        npz_path,
        times=times.astype(np.float64),
        states=states.astype(np.float32),
    )
    meta.sha256 = sha256_file(npz_path)
    meta_path = os.path.join(directory, f"{traj_id}.json")
    with open(meta_path, "w") as f:
        json.dump(meta.to_dict(), f, indent=2)
    return npz_path


def partition_indices(n_total: int, fractions: Tuple[float, float, float, float], seed: int) -> List[str]:
    """Assign indices to train/validation/calibration/test deterministically by seed."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_total)
    n_tr, n_val, n_cal = [
        int(round(f * n_total)) for f in fractions[:3]
    ]
    # Ensure calibration + test at least 1 each when enough data
    n_tr = min(n_tr, n_total - 3)
    n_val = min(n_val, max(0, n_total - n_tr - 2))
    n_cal = min(n_cal, max(0, n_total - n_tr - n_val - 1))
    labels = ["test"] * n_total
    labels[:n_tr] = ["train"] * n_tr
    labels[n_tr:n_tr + n_val] = ["validation"] * n_val
    labels[n_tr + n_val:n_tr + n_val + n_cal] = ["calibration"] * n_cal
    out = [""] * n_total
    for idx, lab in zip(perm, labels):
        out[idx] = lab
    return out


def generate_burgers_dataset(
    output_dir: str,
    n_train: int = 400,
    n_val: int = 100,
    n_cal: int = 100,
    n_test: int = 200,
    n_ood: int = 100,
    nx: int = 64,
    L: float = 1.0,
    nu_train_range: Tuple[float, float] = (0.01, 0.03),
    nu_ood: Tuple[float, float] = (0.005, 0.008),
    t_end: float = 0.5,
    ic_type: str = "fourier",
    base_seed: int = 0,
    safety: float = 0.35,
) -> Dict:
    """Generate a full Burgers dataset with train/val/cal/test/ood partitions.

    Returns the manifest dict (also written to disk).
    """
    root = os.path.join(output_dir, "burgers")
    partitions = ["train", "validation", "calibration", "test", "ood_test"]
    counts = {"train": n_train, "validation": n_val, "calibration": n_cal, "test": n_test, "ood_test": n_ood}

    manifest: Dict = {
        "pde": "burgers",
        "nx": nx,
        "L": L,
        "t_end": t_end,
        "ic_type": ic_type,
        "base_seed": base_seed,
        "partitions": {},
        "trajectories": [],
    }

    traj_counter = 0
    for partition in partitions:
        count = counts[partition]
        pdir = os.path.join(root, partition)
        os.makedirs(pdir, exist_ok=True)
        for i in range(count):
            # Deterministic seed per trajectory, derived from base_seed and counter
            seed = base_seed + 100000 * traj_counter + i
            traj_counter += 1
            if partition == "ood_test":
                nu = float(np.random.default_rng(seed).uniform(*nu_ood))
                ic_mean = float(np.random.default_rng(seed + 1).uniform(-0.5, 0.5))
            else:
                nu = float(np.random.default_rng(seed).uniform(*nu_train_range))
                ic_mean = 0.0
            times, states, meta = generate_burgers_trajectory(
                seed=seed,
                nx=nx,
                L=L,
                nu=nu,
                t_end=t_end,
                ic_type=ic_type,
                ic_mean=ic_mean,
                safety=safety,
            )
            meta.partition = partition
            npz_path = save_trajectory(times, states, meta, pdir)
            manifest["trajectories"].append(meta.to_dict())
        manifest["partitions"][partition] = {
            "directory": pdir,
            "count": count,
        }

    # Write manifest + checksum
    manifest_path = os.path.join(root, "dataset_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    manifest["manifest_sha256"] = sha256_file(manifest_path)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate BRACE-CPU dataset")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--n-train", type=int, default=400)
    parser.add_argument("--n-val", type=int, default=100)
    parser.add_argument("--n-cal", type=int, default=100)
    parser.add_argument("--n-test", type=int, default=200)
    parser.add_argument("--n-ood", type=int, default=100)
    parser.add_argument("--base-seed", type=int, default=0)
    args = parser.parse_args()

    manifest = generate_burgers_dataset(
        output_dir=args.output_dir,
        n_train=args.n_train,
        n_val=args.n_val,
        n_cal=args.n_cal,
        n_test=args.n_test,
        n_ood=args.n_ood,
        nx=args.nx,
        base_seed=args.base_seed,
    )
    print(f"Generated {sum(len(p['trajectories']) for p in [manifest])} trajectories")
    print(f"Manifest: {os.path.join(args.output_dir, 'burgers', 'dataset_manifest.json')}")
