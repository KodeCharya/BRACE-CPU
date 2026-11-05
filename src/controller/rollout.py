"""Rollout runner for selective numerical intervention.

Simulates a rollout under a given controller policy, measuring real wall-clock
cost for every neural inference, every solver invocation, and every controller
(risk/compute) decision. None is free.

Cost accounting (fixing defect CF-1/CF-2):
  - ``decide()`` now returns (intervene, u_pred, elapsed). The elapsed time
    includes the surrogate prediction and risk computation performed inside the
    decision. The rollout reuses ``u_pred`` instead of calling predict twice.
  - Neural inference for policies A and G (non-intervention) is timed separately
    as neural_wall_s.
  - Controller/risk computation for C, D, E, F, H (and G's lookup) is timed as
    controller_wall_s (it subsumes the neural prediction used to assess risk).
  - G's one-time pass-1 preparation time is reported as prepare_wall_s.
  - total_wall_s = neural + solver + controller (+ prepare reported separately).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from solvers.base import relative_l2_error, max_abs_error


@dataclass
class RolloutResult:
    policy: str
    n_steps: int
    intervention_count: int
    intervention_fraction: float
    total_wall_s: float
    neural_wall_s: float
    solver_wall_s: float
    controller_wall_s: float
    prepare_wall_s: float
    solver_calls: int
    # accuracy vs reference
    rel_l2_mean: float
    rel_l2_max: float
    rel_l2_final: float
    max_abs_mean: float
    # per-step arrays
    intervened: np.ndarray = field(repr=False)
    states: np.ndarray = field(repr=False)
    times: np.ndarray = field(repr=False)
    step_errors: np.ndarray = field(repr=False)


def _reference_at(reference_states, reference_times, t: float) -> Optional[np.ndarray]:
    if reference_states is None or reference_times is None:
        return None
    j = int(round((t - reference_times[0]) / (reference_times[1] - reference_times[0])))
    j = min(max(j, 0), reference_states.shape[0] - 1)
    return reference_states[j]


def simulate_rollout(controller, u0: np.ndarray, params: np.ndarray, t0: float,
                     dt: float, n_steps: int,
                     reference_states: Optional[np.ndarray] = None,
                     reference_times: Optional[np.ndarray] = None) -> RolloutResult:
    ctx = controller.ctx
    prepare_s = controller.prepare(u0, params, t0, dt, n_steps,
                                   reference_states, reference_times)

    states = np.empty((n_steps + 1, u0.shape[0]), dtype=np.float64)
    times = np.empty(n_steps + 1, dtype=np.float64)
    intervened = np.zeros(n_steps, dtype=bool)
    step_errors = np.empty(n_steps, dtype=np.float64)

    states[0] = u0
    times[0] = t0
    u = u0.copy()
    t = t0

    neural_s = 0.0
    solver_s = 0.0
    controller_s = controller.controller_time_total  # start at 0; decide accumulates

    for i in range(n_steps):
        intervene, u_pred, dec_elapsed = controller.decide(u, params, t, dt, i)
        controller.record_controller(dec_elapsed)

        if intervene:
            ts = time.perf_counter()
            u = ctx.solver.advance(u, t, dt, 1)   # one-step numerical correction from current state
            te = time.perf_counter()
            controller.record_solver(te - ts)
            solver_s += (te - ts)
            intervened[i] = True
        else:
            if u_pred is not None:
                # decide() already computed u_pred (risk-based policies C,D,E,F,H).
                # Reuse it; the neural+controller cost is in controller_s.
                u = u_pred
            else:
                # No prediction available from decide (A, B, G non-intervention).
                # A needs a neural step; G non-intervention needs a neural step.
                ts = time.perf_counter()
                u = ctx.surrogate.predict(u, params, np.array([t]))
                te = time.perf_counter()
                controller.record_neural(te - ts)
                neural_s += (te - ts)
        states[i + 1] = u
        times[i + 1] = t + dt
        t += dt
        u_ref = _reference_at(reference_states, reference_times, times[i + 1])
        if u_ref is not None:
            step_errors[i] = relative_l2_error(u, u_ref)

    controller_s = controller.controller_time_total
    neural_s = controller.neural_time_total
    solver_s = controller.solver_time_total
    total_s = neural_s + solver_s + controller_s

    result = RolloutResult(
        policy=controller.mode.value,
        n_steps=n_steps,
        intervention_count=int(intervened.sum()),
        intervention_fraction=float(intervened.mean()) if n_steps else 0.0,
        total_wall_s=total_s,
        neural_wall_s=neural_s,
        solver_wall_s=solver_s,
        controller_wall_s=controller_s,
        prepare_wall_s=prepare_s,
        solver_calls=int(intervened.sum()),
        rel_l2_mean=float(np.mean(step_errors)) if step_errors.size else float("nan"),
        rel_l2_max=float(np.max(step_errors)) if step_errors.size else float("nan"),
        rel_l2_final=float(step_errors[-1]) if step_errors.size else float("nan"),
        max_abs_mean=float("nan"),
        intervened=intervened,
        states=states,
        times=times,
        step_errors=step_errors,
    )
    return result


def run_policy(policy, surrogate, solver, risk_estimator, u0, params, t0, dt,
               n_steps, budget: float, threshold: float,
               reference_states=None, reference_times=None, L: float = 1.0,
               target_rate: float | None = None):
    """Convenience: build a controller, run a rollout, return the result."""
    from .policies import Policy, ControlContext, Controller
    if target_rate is None:
        target_rate = budget
    ctx = ControlContext(
        surrogate=surrogate, solver=solver, risk_estimator=risk_estimator,
        L=L, dt=dt, threshold=threshold, budget=budget,
        target_intervention_rate=target_rate,
    )
    controller = Controller(policy, ctx)
    return simulate_rollout(controller, u0, params, t0, dt, n_steps,
                            reference_states, reference_times)


if __name__ == "__main__":
    print("Rollout runner ready.")
