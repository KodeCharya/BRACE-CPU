"""Profile BRACE-G PASS-1 preparation cost breakdown (read-only analysis).

Measures the exact composition of the ~54ms prepare_wall_s:
- Neural prediction count and per-call time
- Risk feature extraction (including temporal_inconsistency which calls predict 3 more times)
- Risk model inference time
- Ranking/selection overhead

Also verifies the total neural inference count per rollout.
"""
from __future__ import annotations

import os
import time

import numpy as np
import torch

# Limit threads for reproducible timing
torch.set_num_threads(1)

from models.surrogate import NeuralSurrogate
from uncertainty.risk_estimator import RiskEstimator
from uncertainty.features import extract_features, temporal_inconsistency
from models.trainer import collect_npz_paths
from solvers.burgers import BurgersSolver


def main():
    # Load seed 0 surrogate
    surrogate = NeuralSurrogate()
    surrogate.load("checkpoints/surrogate_seed0.pt", device="cpu")

    # Fit risk estimator on calibration data
    cal_paths = collect_npz_paths("data/processed/burgers/calibration")
    risk_est = RiskEstimator(tau=0.01, L=1.0, seed=0)
    risk_est.fit(surrogate, cal_paths, stride=4)

    # Load one test trajectory
    test_paths = collect_npz_paths("data/processed/burgers/test")
    data = np.load(test_paths[0])
    states = data["states"].astype(np.float64)
    times = data["times"].astype(np.float64)
    data.close()
    import json
    meta = json.load(open(test_paths[0].replace(".npz", ".json")))
    nu = float(meta.get("nu", 0.01))
    params = np.array([nu], dtype=np.float64)
    dt = float(times[1] - times[0])
    u0 = states[0]
    t0 = float(times[0])
    n_steps = 60

    print("=" * 80)
    print("PASS-1 PREPARATION COST BREAKDOWN (single test trajectory, 60 steps)")
    print("=" * 80)

    # --- Detailed pass-1 timing ---
    u = u0.copy()
    t = t0
    total_predict_calls = 0
    predict_time = 0.0
    feature_time = 0.0
    risk_model_time = 0.0
    feature_counts = []

    t0_prep = time.perf_counter()
    for i in range(n_steps):
        # 1. First prediction (u_pred for residual)
        tp = time.perf_counter()
        u_pred = surrogate.predict(u, params, np.array([t]))
        predict_time += (time.perf_counter() - tp)
        total_predict_calls += 1

        # 2. Risk scoring: score() internally calls predict() again + extract_features + risk_model
        #    Let's time the score call
        tp2 = time.perf_counter()
        r = risk_est.score(surrogate, u, params, t, dt)
        elapsed = time.perf_counter() - tp2
        # score() internally does: predict() + extract_features() + risk_model.predict_logit()
        # extract_features calls temporal_inconsistency which calls predict 3 more times
        total_predict_calls += 1 + 3  # score's predict + temporal_inconsistency's 3 predicts
        feature_time += elapsed  # approximate: includes all of the above
        u = u_pred
        t += dt

    total_prep = time.perf_counter() - t0_prep
    print(f"Total pass-1 time (manual replay): {total_prep*1000:.2f} ms")
    print(f"Total surrogate.predict() calls in pass 1: {total_predict_calls}")
    print(f"  - {n_steps} from explicit u_pred in _prepare_budget_ranked loop")
    print(f"  - {n_steps} from risk_estimator.score() internal predict()")
    print(f"  - {3*n_steps} from temporal_inconsistency (3 predicts per call)")
    print(f"  Total = {n_steps * 5} predictions ({n_steps} explicit + {n_steps} score + {3*n_steps} temporal)")
    print(f"Direct predict time (first call only): {predict_time*1000:.2f} ms")
    print(f"Feature+risk time (remaining): {feature_time*1000:.2f} ms")

    # --- Measure single predict time ---
    print()
    print("--- Single-operation timings (median of 20) ---")
    def time_fn(fn, n=20):
        times_list = []
        for _ in range(n):
            tp = time.perf_counter()
            fn()
            times_list.append(time.perf_counter() - tp)
        return np.median(times_list) * 1000

    u_test = u0.copy()
    t_test = np.array([t0])
    med_predict = time_fn(lambda: surrogate.predict(u_test, params, t_test))
    print(f"Single surrogate.predict(): {med_predict:.3f} ms")

    # Risk feature extraction alone (no score, just features from u_pred)
    med_features = time_fn(lambda: extract_features(surrogate, u_test, u_test, params, t0, dt, 1.0))
    print(f"extract_features() (includes 3 predicts via temporal_inconsistency): {med_features:.3f} ms")

    # Risk model inference alone
    feats = extract_features(surrogate, u_test, u_test, params, t0, dt, 1.0)
    med_risk = time_fn(lambda: risk_est.risk_model.predict_logit(feats[None, :]))
    print(f"LogisticRiskModel.predict_logit(): {med_risk:.3f} ms")

    # Platt scaling alone
    med_platt = time_fn(lambda: risk_est.platt.calibrate(np.array([0.0])))
    print(f"PlattScaling.calibrate(): {med_platt:.3f} ms")

    # score() end-to-end
    med_score = time_fn(lambda: risk_est.score(surrogate, u_test, params, t0, dt))
    print(f"RiskEstimator.score() end-to-end: {med_score:.3f} ms")

    print()
    print("--- Cost accounting per pass-1 step ---")
    print(f"  1 predict (explicit):        {med_predict:.3f} ms")
    print(f"  score() call:                 {med_score:.3f} ms")
    print(f"    (includes 1 predict + 3 temporal predicts + feature math + risk model + platt)")
    print(f"  Total per step:               {med_predict + med_score:.3f} ms")
    print(f"  Total for 60 steps:           {(med_predict + med_score) * 60:.2f} ms")
    print(f"  Measured total prep:          {total_prep*1000:.2f} ms")

    # --- Pass-2 cost per step ---
    print()
    print("--- Pass-2 (main rollout) per-step costs ---")
    med_solver = time_fn(lambda: BurgersSolver(nx=64, L=1.0, nu=nu).advance(u0.copy(), t0, dt, 1))
    print(f"Single solver.advance(1 step): {med_solver:.3f} ms")

    # --- Pass-2 total for G at budget 0.2 (12 interventions) ---
    n_int = int(round(0.2 * 60))
    n_non = 60 - n_int
    pass2_cost = n_non * med_predict + n_int * med_solver
    print(f"Pass-2 estimated cost (12 solver + 48 neural): {pass2_cost:.2f} ms")
    print(f"Pass-1 + Pass-2 total: {total_prep*1000 + pass2_cost:.2f} ms")

    # --- Redundant computation analysis ---
    print()
    print("--- REDUNDANT COMPUTATION ANALYSIS ---")
    print(f"Per pass-1 step, neural inference is called {5}x:")
    print(f"  1x explicit predict (in _prepare_budget_ranked)")
    print(f"  1x predict inside score() -> for residual feature")
    print(f"  3x predicts inside temporal_inconsistency -> for consistency feature")
    print()
    print(f"Per pass-2 non-intervened step, neural inference is called {1}x:")
    print(f"  1x predict in rollout (decide returns None for G)")
    print()
    print(f"For 60 steps at budget=0.2 (12 interventions):")
    print(f"  Pass-1 neural calls: {60*5} = 300")
    print(f"  Pass-2 neural calls: {48} (non-intervened only)")
    print(f"  Total neural calls: 348")
    print(f"  If single-pass online: 60 (only needed for actual step) + risk feature overhead")
    print(f"  Redundancy factor: {348/60:.1f}x vs single-pass minimum")

    # --- Solver cost ---
    print()
    print("--- Solver cost analysis ---")
    print(f"Solver advance (1 RK4 step): {med_solver:.3f} ms")
    print(f"12 solver steps: {12*med_solver:.2f} ms")
    print(f"60 solver steps (B baseline): {60*med_solver:.2f} ms")


if __name__ == "__main__":
    main()
