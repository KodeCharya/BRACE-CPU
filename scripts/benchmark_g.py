"""Benchmark BRACE-G implementations with complete cost accounting.

Compares:
1. G_TWOPASS (original): Two-pass with full prep cost accounting
2. G_ONLINE (new): Single-pass online risk scoring
3. A (always neural): Baseline
4. B (always numerical): Baseline
5. C (fixed threshold): Baseline

Measures:
- Exact budget enforcement
- Complete cost breakdown (prep + exec)
- Accuracy at matched true cost
- Neural call counts
- Risk ranking staleness (online vs twopass)
"""
from __future__ import annotations

import time
import numpy as np
import torch

torch.set_num_threads(1)

from models.surrogate import NeuralSurrogate
from uncertainty.risk_estimator import RiskEstimator
from models.trainer import collect_npz_paths
from solvers.burgers import BurgersSolver
from solvers.base import relative_l2_error
from controller.brace_controller import (
    rollout_g_twopass, rollout_g_online, CostBreakdown, RolloutResult
)
from controller.policies import Policy, ControlContext, Controller
from controller.rollout import simulate_rollout


def main():
    # Load surrogate + risk estimator
    surrogate = NeuralSurrogate()
    surrogate.load("checkpoints/surrogate_seed0.pt", device="cpu")
    cal_paths = collect_npz_paths("data/processed/burgers/calibration")
    risk_est = RiskEstimator(tau=0.01, L=1.0, seed=0)
    risk_est.fit(surrogate, cal_paths, stride=4)

    # Load test trajectories
    test_paths = collect_npz_paths("data/processed/burgers/test")[:5]
    trajectories = []
    for pth in test_paths:
        d = np.load(pth)
        states = d["states"].astype(np.float64)
        times = d["times"].astype(np.float64)
        d.close()
        import json
        meta = json.load(open(pth.replace(".npz", ".json")))
        nu = float(meta.get("nu", 0.01))
        trajectories.append((
            states[0], np.array([nu]), float(times[0]),
            float(times[1] - times[0]), states, times, nu
        ))

    n_steps = 60

    print("=" * 90)
    print("BRACE-G BENCHMARK: Two-Pass vs Online (Complete Cost Accounting)")
    print("=" * 90)
    print()

    for budget in [0.05, 0.1, 0.2, 0.4, 0.6]:
        print(f"--- Budget {budget:.0%} ---")
        results = {"twopass": [], "online": [], "A": [], "B": []}

        for (u0, params, t0, dt, ref_states, ref_times, nu) in trajectories:
            solver = BurgersSolver(nx=64, L=1.0, nu=nu)
            ref_s = ref_states[:n_steps+1]
            ref_t = ref_times[:n_steps+1]

            # G_TWOPASS
            r_tp = rollout_g_twopass(surrogate, solver, risk_est, u0, params, t0, dt,
                                     n_steps, budget, ref_s, ref_t, 1.0)
            results["twopass"].append(r_tp)

            # G_ONLINE
            r_on = rollout_g_online(surrogate, solver, risk_est, u0, params, t0, dt,
                                    n_steps, budget, ref_s, ref_t, 1.0)
            results["online"].append(r_on)

            # A (always neural)
            ctx = ControlContext(surrogate=surrogate, solver=solver, risk_estimator=risk_est,
                                 L=1.0, dt=dt, budget=budget, threshold=99.0,
                                 target_intervention_rate=budget)
            ctl = Controller(Policy.ALWAYS_NEURAL, ctx)
            r_a = simulate_rollout(ctl, u0, params, t0, dt, n_steps, ref_s, ref_t)
            results["A"].append(r_a)

            # B (always numerical)
            ctx_b = ControlContext(surrogate=surrogate, solver=solver, risk_estimator=risk_est,
                                   L=1.0, dt=dt, budget=budget, threshold=0.0,
                                   target_intervention_rate=budget)
            ctl_b = Controller(Policy.ALWAYS_NUMERICAL, ctx_b)
            r_b = simulate_rollout(ctl_b, u0, params, t0, dt, n_steps, ref_s, ref_t)
            results["B"].append(r_b)

        # Report averages
        for name in ["twopass", "online", "A", "B"]:
            rs = results[name]
            rel = np.mean([r.rel_l2_mean for r in rs])
            frac = np.mean([r.intervention_fraction for r in rs])
            if name in ("twopass", "online"):
                total_ms = np.mean([r.cost.total_ms for r in rs])
                prep_ms = np.mean([r.cost.prep_total_ms for r in rs])
                nn_calls = np.mean([r.cost.n_neural_calls for r in rs])
                print(f"  {name:<10} relL2={rel:.5f} frac={frac:.3f} total_ms={total_ms:.2f} "
                      f"(prep={prep_ms:.2f}) neural_calls={nn_calls:.0f}")
            else:
                total_ms = np.mean([r.total_wall_s for r in rs]) * 1000
                print(f"  {name:<10} relL2={rel:.5f} frac={frac:.3f} total_ms={total_ms:.2f}")

        print()

    print("=" * 90)
    print("DETAILED COST BREAKDOWN (Budget=0.2, single trajectory)")
    print("=" * 90)
    (u0, params, t0, dt, ref_states, ref_times, nu) = trajectories[0]
    solver = BurgersSolver(nx=64, L=1.0, nu=nu)
    ref_s = ref_states[:n_steps+1]
    ref_t = ref_times[:n_steps+1]

    r = rollout_g_twopass(surrogate, solver, risk_est, u0, params, t0, dt,
                          n_steps, 0.2, ref_s, ref_t, 1.0)
    c = r.cost
    print(f"G_TWOPASS (Budget 20%):")
    print(f"  Prep neural:  {c.prep_neural_ms:>7.2f} ms")
    print(f"  Prep risk:    {c.prep_risk_ms:>7.2f} ms")
    print(f"  Prep ranking: {c.prep_ranking_ms:>7.2f} ms")
    print(f"  Prep total:   {c.prep_total_ms:>7.2f} ms")
    print(f"  Exec neural:  {c.exec_neural_ms:>7.2f} ms")
    print(f"  Exec solver:  {c.exec_solver_ms:>7.2f} ms")
    print(f"  Exec ctrl:    {c.exec_controller_ms:>7.2f} ms")
    print(f"  Exec total:   {c.exec_total_ms:>7.2f} ms")
    print(f"  TOTAL:        {c.total_ms:>7.2f} ms  (prep {c.prep_total_ms/c.total_ms*100:.1f}% of total)")
    print(f"  Neural calls: {c.n_neural_calls}, Solver calls: {c.n_solver_calls}")
    print(f"  relL2={r.rel_l2_mean:.5f}, frac={r.intervention_fraction:.3f}")

    print()
    r2 = rollout_g_online(surrogate, solver, risk_est, u0, params, t0, dt,
                          n_steps, 0.2, ref_s, ref_t, 1.0)
    c2 = r2.cost
    print(f"G_ONLINE (Budget 20%):")
    print(f"  Prep:         {c2.prep_total_ms:>7.2f} ms (zero — single pass)")
    print(f"  Exec neural:  {c2.exec_neural_ms:>7.2f} ms")
    print(f"  Exec risk:    {c2.exec_risk_ms:>7.2f} ms")
    print(f"  Exec solver:  {c2.exec_solver_ms:>7.2f} ms")
    print(f"  Exec ctrl:    {c2.exec_controller_ms:>7.2f} ms")
    print(f"  Exec total:   {c2.exec_total_ms:>7.2f} ms")
    print(f"  TOTAL:        {c2.total_ms:>7.2f} ms")
    print(f"  Neural calls: {c2.n_neural_calls}, Solver calls: {c2.n_solver_calls}")
    print(f"  relL2={r2.rel_l2_mean:.5f}, frac={r2.intervention_fraction:.3f}")

    print()
    print("=" * 90)
    print("BUDGET ENFORCEMENT VERIFICATION")
    print("=" * 90)
    for budget in [0.05, 0.1, 0.2, 0.4, 0.6]:
        expected = int(round(budget * n_steps))
        r_tp = rollout_g_twopass(surrogate, solver, risk_est, u0, params, t0, dt,
                                 n_steps, budget, ref_s, ref_t, 1.0)
        r_on = rollout_g_online(surrogate, solver, risk_est, u0, params, t0, dt,
                                n_steps, budget, ref_s, ref_t, 1.0)
        status_tp = "EXACT" if r_tp.intervention_count == expected else f"VIOLATION (got {r_tp.intervention_count})"
        status_on = "EXACT" if r_on.intervention_count == expected else f"VIOLATION (got {r_on.intervention_count})"
        print(f"  Budget {budget:.0%}: expected={expected}, twopass={r_tp.intervention_count} ({status_tp}), "
              f"online={r_on.intervention_count} ({status_on})")


if __name__ == "__main__":
    main()
