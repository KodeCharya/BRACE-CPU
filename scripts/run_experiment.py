"""BRACE-CPU experiment harness.

Trains surrogate(s), fits risk estimator(s), runs all control policies (A-H)
on test trajectories at matched intervention budgets, and saves raw results.

Designed for the 8 GiB CPU-only constraint: one surrogate + risk estimator in
memory at a time; experiments run sequentially; results written incrementally.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np

from models.surrogate import NeuralSurrogate, SurrogateConfig
from models.trainer import train_surrogate, collect_npz_paths
from uncertainty.risk_estimator import RiskEstimator
from controller.policies import Policy, ControlContext, Controller
from controller.rollout import run_policy, simulate_rollout
from solvers.burgers import BurgersSolver


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
    return datetime.now(timezone.utc).isoformat()


def load_manifest(data_dir: str) -> dict:
    path = os.path.join(data_dir, "burgers", "dataset_manifest.json")
    with open(path) as f:
        return json.load(f)


def dataset_checksum(data_dir: str) -> str:
    import hashlib
    path = os.path.join(data_dir, "burgers", "dataset_manifest.json")
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_and_save_surrogate(seed: int, train_dir: str, val_dir: str,
                             checkpoint_path: str, nx: int = 64,
                             epochs: int = 40, stride: int = 4) -> NeuralSurrogate:
    cfg = SurrogateConfig(nx=nx, n_params=1, n_time=1, hidden_layers=3,
                          hidden_units=64, activation="relu", lr=1e-3,
                          batch_size=256, epochs=epochs, seed=seed, device="cpu")
    s = NeuralSurrogate(cfg)
    train_surrogate(s, train_dir, val_dir=val_dir, stride=stride, device="cpu")
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    s.save(checkpoint_path)
    return s


def load_surrogate(checkpoint_path: str) -> NeuralSurrogate:
    s = NeuralSurrogate()
    s.load(checkpoint_path, device="cpu")
    return s


# ---------------------------------------------------------------------------
# Risk estimator fitting
# ---------------------------------------------------------------------------

def fit_risk_estimator(surrogate: NeuralSurrogate, cal_dir: str, tau: float = 0.01,
                       L: float = 1.0, stride: int = 4, seed: int = 0) -> RiskEstimator:
    cal_paths = collect_npz_paths(cal_dir)
    re = RiskEstimator(tau=tau, L=L, seed=seed)
    re.fit(surrogate, cal_paths, stride=stride)
    return re


# ---------------------------------------------------------------------------
# Threshold tuning (match intervention rate to a target budget)
# ---------------------------------------------------------------------------

def _calibration_scores(surrogate, risk_est, cal_paths: List[str], mode: str,
                        stride: int, dt: float, L: float) -> np.ndarray:
    """Compute the decision score used by each threshold policy on the
    calibration set (surrogate-only trajectory), for threshold tuning."""
    scores = []
    for path in cal_paths:
        data = np.load(path)
        states = data["states"].astype(np.float64)
        times = data["times"].astype(np.float64)
        data.close()
        meta = json.load(open(path[:-4] + ".json"))
        nu = float(meta.get("nu", 0.01))
        params = np.array([nu], dtype=np.float64)
        dti = float(times[1] - times[0]) if times.size > 1 else dt
        for i in range(0, states.shape[0] - 1, stride):
            u_t = states[i]
            t = float(times[i])
            u_pred = surrogate.predict(u_t, params, np.array([t]))
            if mode == Policy.FIXED_THRESHOLD:
                from uncertainty.features import normalized_residual
                sc = normalized_residual(u_t, u_pred, nu, dti, L)
            elif mode == Policy.UNCALIBRATED_RISK:
                from uncertainty.features import extract_features
                feats = extract_features(surrogate, u_t, u_pred, params, t, dti, L)
                logit = risk_est.risk_model.predict_logit(feats[None, :])[0]
                sc = float(1.0 / (1.0 + np.exp(-logit)))
            elif mode == Policy.CALIBRATED_RISK:
                sc = risk_est.score(surrogate, u_t, params, t, dti)
            else:
                sc = 0.0
            scores.append(sc)
    return np.array(scores)


def tune_threshold(surrogate, risk_est, cal_dir: str, mode: str, target_rate: float,
                   stride: int, dt: float, L: float) -> float:
    """Pick a threshold on the calibration set so the intervention rate ~= target_rate."""
    cal_paths = collect_npz_paths(cal_dir)
    scores = _calibration_scores(surrogate, risk_est, cal_paths, mode, stride, dt, L)
    if len(scores) == 0:
        return 0.5
    # threshold = quantile such that fraction above == target_rate
    q = 1.0 - target_rate
    q = min(max(q, 0.0), 1.0)
    threshold = float(np.quantile(scores, q))
    return threshold


# ---------------------------------------------------------------------------
# Single rollout evaluation
# ---------------------------------------------------------------------------

def evaluate_policy_on_trajectory(surrogate, solver, risk_est, mode: str,
                                  u0: np.ndarray, params: np.ndarray, t0: float,
                                  dt: float, n_steps: int, budget: float,
                                  threshold: float, L: float,
                                  reference_states: np.ndarray,
                                  reference_times: np.ndarray, target_rate: float):
    ctx = ControlContext(surrogate=surrogate, solver=solver, risk_estimator=risk_est,
                         L=L, dt=dt, budget=budget, threshold=threshold,
                         target_intervention_rate=target_rate)
    controller = Controller(mode, ctx)
    return simulate_rollout(controller, u0, params, t0, dt, n_steps,
                            reference_states, reference_times)


# ---------------------------------------------------------------------------
# Result record
# ---------------------------------------------------------------------------

def make_record(experiment_id: str, seed: int, policy: str, budget: float,
                threshold: float, traj_id: str, result, extra: dict) -> dict:
    return {
        "record_id": uuid.uuid4().hex[:12],
        "experiment_id": experiment_id,
        "timestamp": utc_now(),
        "seed": seed,
        "policy": policy,
        "budget": float(budget),
        "threshold": float(threshold),
        "test_trajectory_id": traj_id,
        "n_steps": result.n_steps,
        "intervention_count": result.intervention_count,
        "intervention_fraction": result.intervention_fraction,
        "rel_l2_mean": result.rel_l2_mean,
        "rel_l2_max": result.rel_l2_max,
        "rel_l2_final": result.rel_l2_final,
        "neural_wall_s": result.neural_wall_s,
        "solver_wall_s": result.solver_wall_s,
        "controller_wall_s": result.controller_wall_s,
        "prepare_wall_s": result.prepare_wall_s,
        "total_wall_s": result.total_wall_s,
        "solver_calls": result.solver_calls,
        **extra,
    }


def save_record(record: dict, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    rid = record.get("record_id", uuid.uuid4().hex[:12])
    path = os.path.join(out_dir, f"{rid}.json")
    with open(path, "w") as f:
        json.dump(record, f, indent=2)
    return path


# ---------------------------------------------------------------------------
# Main experiment driver
# ---------------------------------------------------------------------------

def run_experiment_matrix(
    data_dir: str = "data/processed",
    checkpoints_dir: str = "checkpoints",
    results_dir: str = "results/raw",
    seeds: List[int] = [0, 1, 2],
    n_test_trajectories: int = 15,
    n_steps: int = 60,
    budgets: List[float] = [0.05, 0.10, 0.20, 0.40, 0.60],
    tau: float = 0.01,
    L: float = 1.0,
    epochs: int = 40,
    stride: int = 4,
    retrain: bool = True,
) -> List[dict]:
    """Run the full policy matrix across seeds and budgets. Returns all records."""
    train_dir = os.path.join(data_dir, "burgers", "train")
    val_dir = os.path.join(data_dir, "burgers", "validation")
    cal_dir = os.path.join(data_dir, "burgers", "calibration")
    test_dir = os.path.join(data_dir, "burgers", "test")

    test_paths = collect_npz_paths(test_dir)[:n_test_trajectories]
    manifest = load_manifest(data_dir)
    dchecksum = dataset_checksum(data_dir)

    os.makedirs(results_dir, exist_ok=True)
    experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    run_dir = os.path.join(results_dir, experiment_id)
    os.makedirs(run_dir, exist_ok=True)

    # Save experiment config
    config = {
        "experiment_id": experiment_id,
        "seeds": seeds,
        "n_test_trajectories": n_test_trajectories,
        "n_steps": n_steps,
        "budgets": budgets,
        "tau": tau,
        "L": L,
        "epochs": epochs,
        "stride": stride,
        "dataset_checksum": dchecksum,
        "manifest_partitions": {k: v["count"] for k, v in manifest["partitions"].items()},
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    all_records: List[dict] = []

    for seed in seeds:
        ckpt = os.path.join(checkpoints_dir, f"surrogate_seed{seed}.pt")
        if retrain or not os.path.exists(ckpt):
            print(f"[seed {seed}] training surrogate ...")
            s = train_and_save_surrogate(seed, train_dir, val_dir, ckpt,
                                         epochs=epochs, stride=stride)
        else:
            print(f"[seed {seed}] loading checkpoint {ckpt}")
            s = load_surrogate(ckpt)

        print(f"[seed {seed}] fitting risk estimator ...")
        risk_est = fit_risk_estimator(s, cal_dir, tau=tau, L=L, stride=stride, seed=seed)

        # thresholds for rate-matched threshold policies
        thresholds = {}
        for b in budgets:
            thresholds[("fixed", b)] = tune_threshold(s, risk_est, cal_dir,
                                                      Policy.FIXED_THRESHOLD, b,
                                                      stride, 0.001, L)
            thresholds[("uncalibrated", b)] = tune_threshold(s, risk_est, cal_dir,
                                                             Policy.UNCALIBRATED_RISK, b,
                                                             stride, 0.001, L)
            thresholds[("calibrated", b)] = tune_threshold(s, risk_est, cal_dir,
                                                           Policy.CALIBRATED_RISK, b,
                                                           stride, 0.001, L)

        for tidx, tpath in enumerate(test_paths):
            data = np.load(tpath)
            states = data["states"].astype(np.float64)
            times = data["times"].astype(np.float64)
            data.close()
            meta = json.load(open(tpath[:-4] + ".json"))
            nu = float(meta.get("nu", 0.01))
            params = np.array([nu], dtype=np.float64)
            dt = float(times[1] - times[0]) if times.size > 1 else 0.001
            u0 = states[0]
            t0 = float(times[0])
            traj_id = meta["trajectory_id"]

            # solver should match this trajectory's nu
            solver = BurgersSolver(nx=64, L=L, nu=nu)

            n_avail = min(n_steps, states.shape[0] - 1)
            reference_states = states[:n_avail + 1]
            reference_times = times[:n_avail + 1]

            # A: always neural
            res = evaluate_policy_on_trajectory(s, solver, risk_est, Policy.ALWAYS_NEURAL,
                                                u0, params, t0, dt, n_avail, 0.0, 0.0, L,
                                                reference_states, reference_times, 0.0)
            all_records.append(make_record(experiment_id, seed, "A_always_neural", 0.0, 0.0,
                                           traj_id, res, {"nu": nu}))
            save_record(all_records[-1], run_dir)

            # B: always numerical
            res = evaluate_policy_on_trajectory(s, solver, risk_est, Policy.ALWAYS_NUMERICAL,
                                                u0, params, t0, dt, n_avail, 0.0, 0.0, L,
                                                reference_states, reference_times, 0.0)
            all_records.append(make_record(experiment_id, seed, "B_always_numerical", 1.0, 0.0,
                                           traj_id, res, {"nu": nu}))
            save_record(all_records[-1], run_dir)

            # H: oracle
            res = evaluate_policy_on_trajectory(s, solver, risk_est, Policy.ORACLE,
                                                u0, params, t0, dt, n_avail, 0.0, 0.0, L,
                                                reference_states, reference_times, 0.0)
            all_records.append(make_record(experiment_id, seed, "H_oracle", 0.0, 0.0,
                                           traj_id, res, {"nu": nu}))
            save_record(all_records[-1], run_dir)

            # C, D, E, F at matched budgets + G at exact budgets
            for b in budgets:
                # C fixed threshold
                thr = thresholds[("fixed", b)]
                res = evaluate_policy_on_trajectory(s, solver, risk_est, Policy.FIXED_THRESHOLD,
                                                    u0, params, t0, dt, n_avail, b, thr, L,
                                                    reference_states, reference_times, b)
                all_records.append(make_record(experiment_id, seed, "C_fixed_threshold", b, thr,
                                               traj_id, res, {"nu": nu}))
                save_record(all_records[-1], run_dir)

                # D adaptive threshold
                res = evaluate_policy_on_trajectory(s, solver, risk_est, Policy.ADAPTIVE_THRESHOLD,
                                                    u0, params, t0, dt, n_avail, b, thr, L,
                                                    reference_states, reference_times, b)
                all_records.append(make_record(experiment_id, seed, "D_adaptive_threshold", b, thr,
                                               traj_id, res, {"nu": nu}))
                save_record(all_records[-1], run_dir)

                # E uncalibrated risk
                thr = thresholds[("uncalibrated", b)]
                res = evaluate_policy_on_trajectory(s, solver, risk_est, Policy.UNCALIBRATED_RISK,
                                                    u0, params, t0, dt, n_avail, b, thr, L,
                                                    reference_states, reference_times, b)
                all_records.append(make_record(experiment_id, seed, "E_uncalibrated_risk", b, thr,
                                               traj_id, res, {"nu": nu}))
                save_record(all_records[-1], run_dir)

                # F calibrated risk
                thr = thresholds[("calibrated", b)]
                res = evaluate_policy_on_trajectory(s, solver, risk_est, Policy.CALIBRATED_RISK,
                                                    u0, params, t0, dt, n_avail, b, thr, L,
                                                    reference_states, reference_times, b)
                all_records.append(make_record(experiment_id, seed, "F_calibrated_risk", b, thr,
                                               traj_id, res, {"nu": nu}))
                save_record(all_records[-1], run_dir)

                # G budget risk-ranked (BRACE)
                res = evaluate_policy_on_trajectory(s, solver, risk_est, Policy.BUDGET_RISK_RANKED,
                                                    u0, params, t0, dt, n_avail, b, 0.0, L,
                                                    reference_states, reference_times, b)
                all_records.append(make_record(experiment_id, seed, "G_brace_budget", b, 0.0,
                                               traj_id, res, {"nu": nu}))
                save_record(all_records[-1], run_dir)

        # free surrogate before next seed (avoid loading multiple models)
        del s
        del risk_est

    # write combined raw results
    combined = os.path.join(results_dir, f"{experiment_id}_all.json")
    with open(combined, "w") as f:
        json.dump(all_records, f, indent=2)
    print(f"\nExperiment {experiment_id} complete: {len(all_records)} records")
    print(f"Raw results: {run_dir}/ and {combined}")
    return all_records


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run BRACE-CPU experiment matrix")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--results-dir", default="results/raw")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--n-test", type=int, default=15)
    parser.add_argument("--n-steps", type=int, default=60)
    parser.add_argument("--budgets", nargs="+", type=float, default=[0.05, 0.1, 0.2, 0.4, 0.6])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--no-retrain", action="store_true",
                        help="use existing checkpoints if present")
    args = parser.parse_args()

    run_experiment_matrix(
        data_dir=args.data_dir, checkpoints_dir=args.checkpoints_dir,
        results_dir=args.results_dir, seeds=args.seeds,
        n_test_trajectories=args.n_test, n_steps=args.n_steps,
        budgets=args.budgets, epochs=args.epochs, retrain=not args.no_retrain,
    )
