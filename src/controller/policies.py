"""Budget-aware control policies for selective numerical correction.

Modes (per the research protocol):
  A. ALWAYS_NEURAL        -- never invoke the numerical solver
  B. ALWAYS_NUMERICAL     -- always invoke the numerical solver (full fallback)
  C. FIXED_THRESHOLD      -- intervene if normalized residual > fixed threshold
  D. ADAPTIVE_THRESHOLD   -- intervene if residual > threshold adapted to hit a target rate
  E. UNCALIBRATED_RISK    -- intervene if uncalibrated P(failure) > threshold
  F. CALIBRATED_RISK      -- intervene if calibrated P(failure) > threshold
  G. BUDGET_RISK_RANKED   -- within an intervention budget, correct the top-k riskiest steps (BRACE-CPU)
  H. ORACLE               -- intervene where true next-step error > tau (evaluation only)

Every solver invocation is timed and counted; none is treated as free.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Callable, List, Optional

import numpy as np

from solvers.base import relative_l2_error, periodic_gradient, periodic_laplacian
from uncertainty.features import burgers_pde_residual, normalized_residual


def _ref_at(reference_states, reference_times, t: float):
    if reference_states is None or reference_times is None:
        return None
    j = int(round((t - reference_times[0]) / (reference_times[1] - reference_times[0])))
    j = min(max(j, 0), reference_states.shape[0] - 1)
    return reference_states[j]


class Policy(Enum):
    ALWAYS_NEURAL = "always_neural"
    ALWAYS_NUMERICAL = "always_numerical"
    FIXED_THRESHOLD = "fixed_threshold"
    ADAPTIVE_THRESHOLD = "adaptive_threshold"
    UNCALIBRATED_RISK = "uncalibrated_risk"
    CALIBRATED_RISK = "calibrated_risk"
    BUDGET_RISK_RANKED = "budget_risk_ranked"
    ORACLE = "oracle"


@dataclass
class ControlContext:
    """Shared read-mostly context for a rollout."""
    surrogate: object = None
    solver: object = None
    risk_estimator: object = None
    L: float = 1.0
    dt: float = 0.001
    # policy-specific
    threshold: float = 0.02          # residual/risk threshold for C, D, E, F
    budget: float = 0.1              # fraction of steps allowed to intervene (G)
    target_intervention_rate: float = 0.1  # for D
    # precomputed (set by Controller.prepare)
    intervention_indices: Optional[np.ndarray] = None  # for G / H
    # reference trajectory (oracle uses it for lookahead; evaluation only)
    reference_states: Optional[np.ndarray] = None
    reference_times: Optional[np.ndarray] = None


@dataclass
class StepCost:
    neural_time: float = 0.0
    solver_time: float = 0.0
    intervened: bool = False


class Controller:
    def __init__(self, mode: Policy, ctx: ControlContext):
        self.mode = mode
        self.ctx = ctx
        # runtime state
        self.interventions: List[int] = []
        self.neural_time_total = 0.0
        self.solver_time_total = 0.0
        self.controller_time_total = 0.0  # time spent in decide() (risk computation)
        self.prepare_time_total = 0.0     # one-time prep cost (e.g. G pass-1)
        self.n_steps = 0
        # adaptive threshold state (D)
        self.adapt_threshold = ctx.threshold
        self._resid_history: List[float] = []
        self._interv_history: List[float] = []

    # ------------------------------------------------------------------ #
    # Preparation (two-pass policies: G and H)
    # ------------------------------------------------------------------ #
    def prepare(self, u0: np.ndarray, params: np.ndarray, t0: float, dt: float,
                n_steps: int, reference_states: Optional[np.ndarray] = None,
                reference_times: Optional[np.ndarray] = None) -> float:
        """Precompute intervention indices for budget-ranked (G).

        Oracle (H) decides online in ``decide`` and needs no precompute.

        Returns the preparation wall time in seconds.
        """
        import time
        self.ctx.reference_states = reference_states
        self.ctx.reference_times = reference_times
        t0_prep = time.perf_counter()
        if self.mode == Policy.BUDGET_RISK_RANKED:
            self._prepare_budget_ranked(u0, params, t0, dt, n_steps)
        self.n_steps = n_steps
        t1_prep = time.perf_counter()
        self.prepare_time_total = t1_prep - t0_prep
        return self.prepare_time_total

    def _prepare_budget_ranked(self, u0, params, t0, dt, n_steps) -> None:
        """Pass 1: run all-surrogate, score risk at each step, pick top-k by risk."""
        ctx = self.ctx
        k = int(round(ctx.budget * n_steps))
        k = max(0, min(k, n_steps))
        risks = np.zeros(n_steps, dtype=np.float64)
        u = u0.copy()
        t = t0
        for i in range(n_steps):
            u_pred = ctx.surrogate.predict(u, params, np.array([t]))
            r = ctx.risk_estimator.score(ctx.surrogate, u, params, t, dt)
            risks[i] = r
            u = u_pred  # all-surrogate trajectory for risk estimation
            t += dt
        # top-k riskiest step indices (the step i -> i+1 is corrected)
        if k > 0:
            top_idx = np.argsort(risks)[::-1][:k]
            self.ctx.intervention_indices = np.sort(top_idx)
        else:
            self.ctx.intervention_indices = np.array([], dtype=int)

    # ------------------------------------------------------------------ #
    # Per-step decision
    # ------------------------------------------------------------------ #
    def decide(self, u_t: np.ndarray, params: np.ndarray, t: float, dt: float,
               step_idx: int):
        """Decide whether to intervene at this step.

        Returns
        -------
        intervene : bool
        u_pred : np.ndarray or None
            The surrogate's next-state prediction, computed as a side effect of
            risk/residual evaluation. Returned so the rollout can reuse it
            instead of calling ``predict`` a second time. None when no
            prediction is needed (A, B, G).
        elapsed : float
            Wall-clock seconds spent in this decision (risk/controller compute).
        """
        import time
        ctx = self.ctx
        t0 = time.perf_counter()

        if self.mode == Policy.ALWAYS_NEURAL:
            return False, None, time.perf_counter() - t0
        if self.mode == Policy.ALWAYS_NUMERICAL:
            return True, None, time.perf_counter() - t0
        if self.mode == Policy.BUDGET_RISK_RANKED:
            return int(step_idx) in set(ctx.intervention_indices.tolist()), None, time.perf_counter() - t0
        if self.mode == Policy.ORACLE:
            u_pred = ctx.surrogate.predict(u_t, params, np.array([t]))
            u_true_next = _ref_at(ctx.reference_states, ctx.reference_times, t + ctx.dt)
            if u_true_next is None:
                return False, u_pred, time.perf_counter() - t0
            err = relative_l2_error(u_pred, u_true_next)
            return bool(err > getattr(ctx.risk_estimator, "tau", 0.01)), u_pred, time.perf_counter() - t0

        # Need a risk/residual signal for C, D, E, F: compute u_pred once here.
        u_pred = ctx.surrogate.predict(u_t, params, np.array([t]))
        nu = float(params[0]) if params.size else 0.01
        res = normalized_residual(u_t, u_pred, nu, dt, ctx.L)

        if self.mode == Policy.FIXED_THRESHOLD:
            return bool(res > ctx.threshold), u_pred, time.perf_counter() - t0
        if self.mode == Policy.ADAPTIVE_THRESHOLD:
            self._resid_history.append(res)
            decision = bool(res > self.adapt_threshold)
            self._adapt_threshold(decision)
            return decision, u_pred, time.perf_counter() - t0
        if self.mode == Policy.UNCALIBRATED_RISK:
            from uncertainty.features import extract_features
            feats = extract_features(ctx.surrogate, u_t, u_pred, params, t, dt, ctx.L)
            logit = ctx.risk_estimator.risk_model.predict_logit(feats[None, :])[0]
            p = float(1.0 / (1.0 + np.exp(-logit)))
            return bool(p > ctx.threshold), u_pred, time.perf_counter() - t0
        if self.mode == Policy.CALIBRATED_RISK:
            p = ctx.risk_estimator.score(ctx.surrogate, u_t, params, t, dt)
            return bool(p > ctx.threshold), u_pred, time.perf_counter() - t0

        return False, u_pred, time.perf_counter() - t0

    def _adapt_threshold(self, intervened: bool) -> None:
        """Simple proportional adaptation to hit target_intervention_rate (D)."""
        self._interv_history.append(1.0 if intervened else 0.0)
        if len(self._interv_history) < 5:
            return
        recent_rate = float(np.mean(self._interv_history[-20:]))
        target = self.ctx.target_intervention_rate
        error = recent_rate - target
        self.adapt_threshold *= (1.0 + 0.1 * error)
        self.adapt_threshold = float(np.clip(self.adapt_threshold, 1e-4, 1.0))

    # ------------------------------------------------------------------ #
    # Cost recording
    # ------------------------------------------------------------------ #
    def record_neural(self, elapsed: float) -> None:
        self.neural_time_total += elapsed

    def record_controller(self, elapsed: float) -> None:
        self.controller_time_total += elapsed

    def record_solver(self, elapsed: float) -> None:
        self.solver_time_total += elapsed

    @property
    def intervention_count(self) -> int:
        return len(self.interventions)
