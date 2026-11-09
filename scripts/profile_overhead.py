"""Profile numerical solver and torch threading overhead.

Measures:
- Single RK4 step time at different thread counts
- Neural predict overhead at different thread counts  
- NumPy operation overhead (roll-based derivatives)
- Batch vs single inference comparison
"""
from __future__ import annotations

import time
import numpy as np
import torch

from models.surrogate import NeuralSurrogate
from solvers.burgers import BurgersSolver


def time_median(fn, n=50):
    times = []
    for _ in range(n):
        tp = time.perf_counter()
        fn()
        times.append(time.perf_counter() - tp)
    return np.median(times) * 1000


def main():
    surrogate = NeuralSurrogate()
    surrogate.load("checkpoints/surrogate_seed0.pt", device="cpu")

    # Prepare test inputs
    u = np.random.randn(64)
    params = np.array([0.02])
    t = np.array([0.0])
    solver = BurgersSolver(nx=64, L=1.0, nu=0.02)

    print("=" * 80)
    print("THREADING SENSITIVITY ANALYSIS")
    print("=" * 80)

    for nthreads in [1, 2, 4]:
        torch.set_num_threads(nthreads)
        med_predict = time_median(lambda: surrogate.predict(u, params, t), 30)
        med_solver = time_median(lambda: solver.advance(u.copy(), 0.0, 0.001, 1), 30)
        print(f"Threads={nthreads}: predict={med_predict:.3f}ms solver_step={med_solver:.3f}ms")

    print()
    print("=" * 80)
    print("BATCH vs SINGLE INFERENCE")
    print("=" * 80)

    # Batch of 60 predictions (all at once)
    u_batch = np.random.randn(60, 64)
    p_batch = np.tile(params, (60, 1))
    t_batch = np.arange(60) * 0.001

    torch.set_num_threads(4)
    med_batch = time_median(lambda: surrogate.predict(u_batch, p_batch, t_batch), 20)
    med_single_x60 = time_median(lambda: [surrogate.predict(u_batch[i], params, np.array([t_batch[i]])) for i in range(60)], 20)

    print(f"Batch 60 predictions: {med_batch:.3f} ms")
    print(f"60 single predictions: {med_single_x60:.3f} ms")
    print(f"Batch speedup: {med_single_x60/med_batch:.2f}x")

    print()
    print("=" * 80)
    print("NUMPY DERIVATIVE OVERHEAD (solver rhs internals)")
    print("=" * 80)

    from solvers.base import periodic_gradient, periodic_laplacian
    med_grad = time_median(lambda: periodic_gradient(u, 1.0/64), 100)
    med_lap = time_median(lambda: periodic_laplacian(u, 1.0/64), 100)
    print(f"periodic_gradient: {med_grad:.4f} ms")
    print(f"periodic_laplacian: {med_lap:.4f} ms")

    med_rhs = time_median(lambda: solver.rhs(u, 0.0), 100)
    print(f"BurgersSolver.rhs(): {med_rhs:.4f} ms")
    print(f"  (rhs = flux computation + gradient + laplacian)")

    med_rk4 = time_median(lambda: solver.rk4_step(u.copy(), 0.0, 0.001), 100)
    print(f"rk4_step (4x rhs): {med_rk4:.4f} ms")

    print()
    print("=" * 80)
    print("FEATURE EXTRACTION OVERHEAD BREAKDOWN")
    print("=" * 80)

    from uncertainty.features import (
        normalized_residual, temporal_inconsistency,
        mass_conservation_deviation, increment_magnitude, gradient_magnitude,
    )
    u_pred = u + 0.01 * np.random.randn(64)

    med_res = time_median(lambda: normalized_residual(u, u_pred, 0.02, 0.001, 1.0), 50)
    print(f"normalized_residual: {med_res:.4f} ms (no predict calls)")

    med_temp = time_median(lambda: temporal_inconsistency(surrogate, u, params, 0.0, 0.001), 50)
    print(f"temporal_inconsistency: {med_temp:.4f} ms (3 predict calls)")

    med_mass = time_median(lambda: mass_conservation_deviation(u, u_pred), 50)
    print(f"mass_conservation_deviation: {med_mass:.4f} ms (no predict calls)")

    med_inc = time_median(lambda: increment_magnitude(u, u_pred), 50)
    print(f"increment_magnitude: {med_inc:.4f} ms (no predict calls)")

    med_gradf = time_median(lambda: gradient_magnitude(u, 1.0), 50)
    print(f"gradient_magnitude: {med_gradf:.4f} ms (no predict calls)")

    print()
    print("=" * 80)
    print("TORCH-SPECIFIC OVERHEAD")
    print("=" * 80)

    # Time just the normalization + model call (no predict wrapper overhead)
    from models.mlp import build_model
    model = surrogate.model
    model.eval()
    with torch.no_grad():
        un = torch.randn(1, 64, dtype=torch.float32)
        pn = torch.randn(1, 1, dtype=torch.float32)
        tn = torch.randn(1, 1, dtype=torch.float32)
        med_model = time_median(lambda: model(un, pn, tn), 50)
        print(f"Raw model forward (normalized inputs): {med_model:.4f} ms")

    # One-shot predict time includes: normalize (numpy ops) + model + denormalize
    med_predict_full = time_median(lambda: surrogate.predict(u, params, t), 50)
    print(f"Full predict wrapper: {med_predict_full:.4f} ms")
    print(f"Wrapper overhead (predict - raw model): {med_predict_full - med_model:.4f} ms")


if __name__ == "__main__":
    main()
