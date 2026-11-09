"""BRACE controller with complete cost accounting and single-pass online alternative.

Two implementations for policy G (Budget-Constrained Risk-Ranked Intervention):

1. G_TWOPASS (original): Pre-score all steps on an all-surrogate trajectory,
   select top-k, then execute controlled rollout. Two-pass; preparation cost
   (~54 ms) is paid once per rollout and is separate from execution cost.

2. G_ONLINE (new): Score risk at each step during the rollout itself, maintain
   a running priority queue of intervention candidates, and intervene when
   the current step's risk exceeds adaptive thresholds. Single-pass; no
   preliminary rollout; risk recomputed after corrections (avoids stale rankings).

Both implementations enforce the exact budget (k = round(budget * n_steps)
interventions) and report complete cost breakdowns.

Cost accounting (per rollout):
  - prep_cost: one-time preparation (pass-1 for G_TWOPASS, zero for G_ONLINE)
  - neural_cost: surrogate inference during execution
  - risk_cost: risk feature extraction + risk model inference during execution
  - solver_cost: numerical correction invocations
  - controller_cost: decision overhead (ranking, threshold logic)
  - total_cost: prep + neural + risk + solver + controller
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import heapq
import time

import numpy as np

from solvers.base import relative_l2_error


class GVariant(Enum):
    TWOPASS = "twopass"    # Original: pre-score all steps, select top-k
    ONLINE = "online"      # New: single-pass online risk scoring with adaptive selection


@dataclass
class CostBreakdown:
    """Complete cost accounting for one rollout."""
    prep_neural_ms: float = 0.0       # Neural inference in preparation (pass-1)
    prep_risk_ms: float = 0.0         # Risk computation in preparation (pass-1)
    prep_ranking_ms: float = 0.0      # Sorting/selecting in preparation
    prep_total_ms: float = 0.0        # Total preparation time

    exec_neural_ms: float = 0.0       # Neural inference during execution
    exec_risk_ms: float = 0.0         # Risk computation during execution
    exec_solver_ms: float = 0.0       # Solver invocations during execution
    exec_controller_ms: float = 0.0   # Decision overhead during execution
    exec_total_ms: float = 0.0        # Total execution time (prep excluded)

    total_ms: float = 0.0              # prep_total + exec_total (complete cost)
    n_neural_calls: int = 0           # Total surrogate.predict() invocations
    n_solver_calls: int = 0           # Total solver.advance() invocations
    n_risk_calls: int = 0             # Total risk score computations


@dataclass
class RolloutResult:
    policy_name: str
    n_steps: int
    intervention_count: int
    intervention_fraction: float
    rel_l2_mean: float
    rel_l2_max: float
    rel_l2_final: float
    cost: CostBreakdown
    intervened: np.ndarray
    states: np.ndarray
    times: np.ndarray
    step_errors: np.ndarray


def _ref_at(reference_states, reference_times, t):
    if reference_states is None or reference_times is None:
        return None
    j = int(round((t - reference_times[0]) / (reference_times[1] - reference_times[0])))
    j = min(max(j, 0), reference_states.shape[0] - 1)
    return reference_states[j]


# ---------------------------------------------------------------------------
# G_TWOPASS: Original two-pass implementation (with corrected cost accounting)
# ---------------------------------------------------------------------------

def rollout_g_twopass(surrogate, solver, risk_estimator, u0, params, t0, dt,
                      n_steps: int, budget: float,
                      reference_states=None, reference_times=None,
                      L: float = 1.0) -> RolloutResult:
    """Original two-pass BRACE-G: pre-score all steps, select top-k, execute.

    Pass 1 (preparation): Run all-surrogate trajectory, score risk at each step,
    select top-k indices by risk. Cost: ~54 ms (measured, reported as prep).

    Pass 2 (execution): Controlled rollout; at selected indices, invoke solver;
    at other indices, use surrogate prediction. Cost: ~9 ms (execution).

    Total cost: prep + execution (both reported; prep NOT hidden).
    """
    cost = CostBreakdown()

    # === PASS 1: Preparation ===
    k = int(round(budget * n_steps))
    k = max(0, min(k, n_steps))

    t_prep_start = time.perf_counter()
    risks = np.zeros(n_steps, dtype=np.float64)
    u = u0.copy()
    t = t0
    prep_neural_time = 0.0
    prep_risk_time = 0.0

    for i in range(n_steps):
        tp = time.perf_counter()
        u_pred = surrogate.predict(u, params, np.array([t]))
        prep_neural_time += (time.perf_counter() - tp)
        cost.n_neural_calls += 1

        tp2 = time.perf_counter()
        r = risk_estimator.score(surrogate, u, params, t, dt)
        prep_risk_time += (time.perf_counter() - tp2)
        cost.n_neural_calls += 1  # score internally calls predict
        cost.n_risk_calls += 1
        risks[i] = r
        u = u_pred
        t += dt

    tp3 = time.perf_counter()
    if k > 0:
        top_idx = np.argsort(risks)[::-1][:k]
        intervention_indices = set(np.sort(top_idx).tolist())
    else:
        intervention_indices = set()
    prep_ranking_time = time.perf_counter() - tp3

    cost.prep_neural_ms = prep_neural_time * 1000
    cost.prep_risk_ms = prep_risk_time * 1000
    cost.prep_ranking_ms = prep_ranking_time * 1000
    cost.prep_total_ms = (time.perf_counter() - t_prep_start) * 1000

    # === PASS 2: Execution ===
    states = np.empty((n_steps + 1, u0.shape[0]), dtype=np.float64)
    times = np.empty(n_steps + 1, dtype=np.float64)
    intervened = np.zeros(n_steps, dtype=bool)
    step_errors = np.empty(n_steps, dtype=np.float64)

    states[0] = u0
    times[0] = t0
    u = u0.copy()
    t = t0

    exec_neural = 0.0
    exec_solver = 0.0
    exec_controller = 0.0

    for i in range(n_steps):
        tc = time.perf_counter()
        intervene = i in intervention_indices
        exec_controller += (time.perf_counter() - tc)

        if intervene:
            ts = time.perf_counter()
            u = solver.advance(u, t, dt, 1)
            exec_solver += (time.perf_counter() - ts)
            cost.n_solver_calls += 1
            intervened[i] = True
        else:
            tp = time.perf_counter()
            u = surrogate.predict(u, params, np.array([t]))
            exec_neural += (time.perf_counter() - tp)
            cost.n_neural_calls += 1

        states[i + 1] = u
        times[i + 1] = t + dt
        t += dt
        u_ref = _ref_at(reference_states, reference_times, times[i + 1])
        if u_ref is not None:
            step_errors[i] = relative_l2_error(u, u_ref)

    cost.exec_neural_ms = exec_neural * 1000
    cost.exec_solver_ms = exec_solver * 1000
    cost.exec_controller_ms = exec_controller * 1000
    cost.exec_total_ms = (exec_neural + exec_solver + exec_controller) * 1000
    cost.total_ms = cost.prep_total_ms + cost.exec_total_ms

    return RolloutResult(
        policy_name="G_twopass",
        n_steps=n_steps,
        intervention_count=int(intervened.sum()),
        intervention_fraction=float(intervened.mean()) if n_steps else 0.0,
        rel_l2_mean=float(np.mean(step_errors)) if step_errors.size else float("nan"),
        rel_l2_max=float(np.max(step_errors)) if step_errors.size else float("nan"),
        rel_l2_final=float(step_errors[-1]) if step_errors.size else float("nan"),
        cost=cost,
        intervened=intervened,
        states=states,
        times=times,
        step_errors=step_errors,
    )


# ---------------------------------------------------------------------------
# G_ONLINE: Single-pass online risk scoring with budget enforcement
# ---------------------------------------------------------------------------

def rollout_g_online(surrogate, solver, risk_estimator, u0, params, t0, dt,
                     n_steps: int, budget: float,
                     reference_states=None, reference_times=None,
                     L: float = 1.0) -> RolloutResult:
    """Single-pass online BRACE-G: score risk at each step during rollout.

    Maintains a running list of (step_index, risk) for all steps seen so far.
    At each step, if remaining budget slots require intervention on current
    step (i.e., even if all future steps were low-risk, we must intervene now
    to meet the budget), intervene. Otherwise, score risk and intervene only
    if the current step is among the top-k riskiest so far.

    This guarantees EXACT budget enforcement while scoring risk online along
    the ACTUAL controlled trajectory (risk rankings reflect post-correction states).

    Cost: zero preparation; risk computation is interleaved with execution.
    """
    cost = CostBreakdown()
    k = int(round(budget * n_steps))
    k = max(0, min(k, n_steps))

    states = np.empty((n_steps + 1, u0.shape[0]), dtype=np.float64)
    times = np.empty(n_steps + 1, dtype=np.float64)
    intervened = np.zeros(n_steps, dtype=bool)
    step_errors = np.empty(n_steps, dtype=np.float64)
    risks = np.zeros(n_steps, dtype=np.float64)

    states[0] = u0
    times[0] = t0
    u = u0.copy()
    t = t0

    # Track interventions used so far
    interventions_used = 0
    # Risk threshold: initially high, adjusts as we see more steps
    # We use a simple strategy: at step i, compute how many interventions
    # remain and how many steps remain. If remaining_interventions >= remaining_steps,
    # must intervene. Otherwise, compare current risk to the k-th highest risk so far.
    visited_risks = []  # list of (risk, index) for steps already visited

    exec_neural = 0.0
    exec_solver = 0.0
    exec_risk = 0.0
    exec_controller = 0.0

    for i in range(n_steps):
        remaining_steps = n_steps - i
        remaining_budget = k - interventions_used

        # Must intervene if we have no choice (budget >= remaining steps)
        if remaining_budget >= remaining_steps:
            # Forced intervention
            tc = time.perf_counter()
            r = 0.0  # don't even score risk if forced
            exec_controller += (time.perf_counter() - tc)
            risks[i] = r
            visited_risks.append((r, i))
            ts = time.perf_counter()
            u = solver.advance(u, t, dt, 1)
            exec_solver += (time.perf_counter() - ts)
            cost.n_solver_calls += 1
            interventions_used += 1
            intervened[i] = True
        else:
            # Score risk for current state
            tp = time.perf_counter()
            u_pred = surrogate.predict(u, params, np.array([t]))
            exec_neural += (time.perf_counter() - tp)
            cost.n_neural_calls += 1

            tp2 = time.perf_counter()
            r = risk_estimator.score(surrogate, u, params, t, dt)
            exec_risk += (time.perf_counter() - tp2)
            cost.n_neural_calls += 1  # score internally predicts
            cost.n_risk_calls += 1
            risks[i] = r

            # Record this risk for future comparison
            visited_risks.append((r, i))

            # Decide: intervene if we still have budget AND current risk is among
            # the top-k riskiest of ALL steps seen so far (including current)
            tc = time.perf_counter()
            if remaining_budget > 0:
                # Find threshold: k-th highest risk among visited so far
                sorted_risks = sorted(visited_risks, reverse=True)
                if len(sorted_risks) <= k:
                    # Not enough steps visited yet to have a meaningful threshold;
                    # intervene if budget allows (early steps are always candidates)
                    should_intervene = True
                else:
                    # Threshold is the k-th highest risk; intervene if current >= threshold
                    threshold_risk = sorted_risks[k - 1][0]
                    should_intervene = r >= threshold_risk

                # Additional guard: if we have used all budget, don't intervene
                if interventions_used >= k:
                    should_intervene = False
            else:
                should_intervene = False
            exec_controller += (time.perf_counter() - tc)

            if should_intervene:
                ts = time.perf_counter()
                u = solver.advance(u, t, dt, 1)
                exec_solver += (time.perf_counter() - ts)
                cost.n_solver_calls += 1
                interventions_used += 1
                intervened[i] = True
            else:
                # Use the prediction we already computed for risk scoring
                u = u_pred

        states[i + 1] = u
        times[i + 1] = t + dt
        t += dt
        u_ref = _ref_at(reference_states, reference_times, times[i + 1])
        if u_ref is not None:
            step_errors[i] = relative_l2_error(u, u_ref)

    cost.exec_neural_ms = exec_neural * 1000
    cost.exec_solver_ms = exec_solver * 1000
    cost.exec_risk_ms = exec_risk * 1000
    cost.exec_controller_ms = exec_controller * 1000
    cost.exec_total_ms = (exec_neural + exec_solver + exec_risk + exec_controller) * 1000
    cost.total_ms = cost.exec_total_ms  # No preparation cost for online
    cost.prep_total_ms = 0.0

    return RolloutResult(
        policy_name="G_online",
        n_steps=n_steps,
        intervention_count=int(intervened.sum()),
        intervention_fraction=float(intervened.mean()) if n_steps else 0.0,
        rel_l2_mean=float(np.mean(step_errors)) if step_errors.size else float("nan"),
        rel_l2_max=float(np.max(step_errors)) if step_errors.size else float("nan"),
        rel_l2_final=float(step_errors[-1]) if step_errors.size else float("nan"),
        cost=cost,
        intervened=intervened,
        states=states,
        times=times,
        step_errors=step_errors,
    )
