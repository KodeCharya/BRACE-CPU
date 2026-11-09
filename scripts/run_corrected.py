"""Full experiment with corrected cost accounting and G_ONLINE comparison.

Runs all 8 policies (A-H) plus G_ONLINE on the authoritative test set,
with complete per-rollout cost breakdown and exact budget enforcement
verification. Results written to a NEW directory (never overwrites existing).
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime

import numpy as np
import torch

torch.set_num_threads(1)

from models.surrogate import NeuralSurrogate
from models.trainer import collect_npz_paths
from uncertainty.risk_estimator import RiskEstimator
from solvers.burgers import BurgersSolver
from solvers.base import relative_l2_error
from controller.policies import Policy, ControlContext, Controller
from controller.rollout import simulate_rollout
from controller.brace_controller import rollout_g_twopass, rollout_g_online
from run_experiment import tune_threshold


def run_full_corrected_experiment(
    seeds=[0, 1, 2, 3, 4],
    n_test=25,
    n_steps=60,
    budgets=[0.05, 0.1, 0.2, 0.4, 0.6],
    out_dir="results/rebuild/",
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(out_dir, f"corrected_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    test_paths = collect_npz_paths("data/processed/burgers/test")[:n_test]
    cal_paths = collect_npz_paths("data/processed/burgers/calibration")

    records = []

    for seed in seeds:
        print(f"\n[Seed {seed}] Loading surrogate...")
        surrogate = NeuralSurrogate()
        surrogate.load(f"checkpoints/surrogate_seed{seed}.pt", device="cpu")

        print(f"[Seed {seed}] Fitting risk estimator...")
        risk_est = RiskEstimator(tau=0.01, L=1.0, seed=seed)
        risk_est.fit(surrogate, cal_paths, stride=4)

        # Tune thresholds for C, E, F on calibration set (matches original experiment)
        cal_dir = "data/processed/burgers/calibration"
        thresholds = {}
        for bud in budgets:
            thresholds[("fixed", bud)] = tune_threshold(
                surrogate, risk_est, cal_dir, Policy.FIXED_THRESHOLD, bud, 4, 0.001, 1.0)
            thresholds[("uncalibrated", bud)] = tune_threshold(
                surrogate, risk_est, cal_dir, Policy.UNCALIBRATED_RISK, bud, 4, 0.001, 1.0)
            thresholds[("calibrated", bud)] = tune_threshold(
                surrogate, risk_est, cal_dir, Policy.CALIBRATED_RISK, bud, 4, 0.001, 1.0)
        print(f"[Seed {seed}] Thresholds tuned.")

        for traj_idx, tpath in enumerate(test_paths):
            data = np.load(tpath)
            states = data["states"].astype(np.float64)
            times = data["times"].astype(np.float64)
            data.close()
            meta = json.load(open(tpath.replace(".npz", ".json")))
            nu = float(meta.get("nu", 0.01))
            params = np.array([nu])
            dt = float(times[1] - times[0]) if len(times) > 1 else 0.001
            u0 = states[0]
            t0 = float(times[0])
            traj_id = meta["trajectory_id"]
            ref_states = states[:n_steps+1]
            ref_times = times[:n_steps+1]
            solver = BurgersSolver(nx=64, L=1.0, nu=nu)

            def record(policy_name, budget, result=None, cost_dict=None):
                rec = {
                    "seed": seed,
                    "traj_id": traj_id,
                    "policy": policy_name,
                    "budget": budget,
                    "nu": nu,
                    "timestamp": timestamp,
                }
                if cost_dict:
                    rec.update(cost_dict)
                if result is not None:
                    rec["rel_l2_mean"] = result.rel_l2_mean
                    rec["rel_l2_max"] = result.rel_l2_max
                    rec["intervention_count"] = result.intervention_count
                    rec["intervention_fraction"] = result.intervention_fraction
                    rec["total_wall_s"] = result.total_wall_s
                    rec["neural_wall_s"] = result.neural_wall_s
                    rec["solver_wall_s"] = result.solver_wall_s
                    rec["controller_wall_s"] = result.controller_wall_s
                    rec["prepare_wall_s"] = result.prepare_wall_s
                records.append(rec)
                return rec

            # A: Always neural
            ctx = ControlContext(surrogate=surrogate, solver=solver, risk_estimator=risk_est,
                                 L=1.0, dt=dt, budget=0.0, threshold=99.0,
                                 target_intervention_rate=0.0)
            r = simulate_rollout(Controller(Policy.ALWAYS_NEURAL, ctx), u0, params, t0, dt,
                                 n_steps, ref_states, ref_times)
            record("A", 0.0, r)

            # B: Always numerical
            ctx = ControlContext(surrogate=surrogate, solver=solver, risk_estimator=risk_est,
                                 L=1.0, dt=dt, budget=1.0, threshold=0.0,
                                 target_intervention_rate=1.0)
            r = simulate_rollout(Controller(Policy.ALWAYS_NUMERICAL, ctx), u0, params, t0, dt,
                                 n_steps, ref_states, ref_times)
            record("B", 1.0, r)

            # C, E, F, G_TWOPASS, G_ONLINE at each budget
            for bud in budgets:
                # C: Fixed threshold (tuned threshold)
                thr = thresholds[("fixed", bud)]
                ctx = ControlContext(surrogate=surrogate, solver=solver, risk_estimator=risk_est,
                                     L=1.0, dt=dt, budget=bud, threshold=thr,
                                     target_intervention_rate=bud)
                r = simulate_rollout(Controller(Policy.FIXED_THRESHOLD, ctx), u0, params, t0, dt,
                                     n_steps, ref_states, ref_times)
                record("C", bud, r)

                # E: Uncalibrated risk (tuned threshold)
                thr = thresholds[("uncalibrated", bud)]
                ctx = ControlContext(surrogate=surrogate, solver=solver, risk_estimator=risk_est,
                                     L=1.0, dt=dt, budget=bud, threshold=thr,
                                     target_intervention_rate=bud)
                r = simulate_rollout(Controller(Policy.UNCALIBRATED_RISK, ctx), u0, params, t0, dt,
                                     n_steps, ref_states, ref_times)
                record("E", bud, r)

                # F: Calibrated risk (tuned threshold)
                thr = thresholds[("calibrated", bud)]
                ctx = ControlContext(surrogate=surrogate, solver=solver, risk_estimator=risk_est,
                                     L=1.0, dt=dt, budget=bud, threshold=thr,
                                     target_intervention_rate=bud)
                r = simulate_rollout(Controller(Policy.CALIBRATED_RISK, ctx), u0, params, t0, dt,
                                     n_steps, ref_states, ref_times)
                record("F", bud, r)

                # G_TWOPASS: Budget-constrained (two-pass)
                r = rollout_g_twopass(surrogate, solver, risk_est, u0, params, t0, dt,
                                      n_steps, bud, ref_states, ref_times, 1.0)
                cost_dict = {
                    "true_total_ms": r.cost.total_ms,
                    "prep_ms": r.cost.prep_total_ms,
                    "prep_neural_ms": r.cost.prep_neural_ms,
                    "prep_risk_ms": r.cost.prep_risk_ms,
                    "exec_total_ms": r.cost.exec_total_ms,
                    "n_neural_calls": r.cost.n_neural_calls,
                    "n_solver_calls": r.cost.n_solver_calls,
                    "rel_l2_mean": r.rel_l2_mean,
                    "rel_l2_max": r.rel_l2_max,
                    "intervention_count": r.intervention_count,
                    "intervention_fraction": r.intervention_fraction,
                }
                record("G_TWOPASS", bud, cost_dict=cost_dict)

                # G_ONLINE: Single-pass online (NEW)
                r = rollout_g_online(surrogate, solver, risk_est, u0, params, t0, dt,
                                     n_steps, bud, ref_states, ref_times, 1.0)
                cost_dict = {
                    "true_total_ms": r.cost.total_ms,
                    "prep_ms": r.cost.prep_total_ms,
                    "exec_neural_ms": r.cost.exec_neural_ms,
                    "exec_risk_ms": r.cost.exec_risk_ms,
                    "exec_solver_ms": r.cost.exec_solver_ms,
                    "n_neural_calls": r.cost.n_neural_calls,
                    "n_solver_calls": r.cost.n_solver_calls,
                    "rel_l2_mean": r.rel_l2_mean,
                    "rel_l2_max": r.rel_l2_max,
                    "intervention_count": r.intervention_count,
                    "intervention_fraction": r.intervention_fraction,
                }
                record("G_ONLINE", bud, cost_dict=cost_dict)

            if (traj_idx + 1) % 5 == 0:
                print(f"  Seed {seed}: trajectory {traj_idx+1}/{n_test} done")

    # Save results
    out_path = os.path.join(run_dir, f"corrected_{timestamp}_all.json")
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nSaved {len(records)} records to {out_path}")
    return out_path


if __name__ == "__main__":
    run_full_corrected_experiment()
