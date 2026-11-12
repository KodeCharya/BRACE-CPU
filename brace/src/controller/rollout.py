"""Closed-loop surrogate rollouts with optional budgeted numerical correction."""
from __future__ import annotations

import time

import numpy as np

from ..correction.numerical import Corrector
from ..uncertainty.features import compute_features, ensemble_predict, new_history, step_error


def _mean(preds: np.ndarray) -> np.ndarray:
    return preds.mean(axis=0)


def _filter_solver_params(solver, params: dict) -> dict:
    allowed = {"BurgersSolver": ("nu",), "AllenCahnSolver": ("eps",),
               "AdvectionDiffusionSolver": ("nu", "cx", "cy")}
    keys = allowed.get(type(solver).__name__, tuple(params.keys()))
    return {k: params[k] for k in keys if k in params}


def neural_pass(models, u_ref: np.ndarray, solver, params: dict, dt: float,
                risk_model=None, device: str = "cpu", store_features: bool = True):
    """Autoregressive ensemble rollout with no correction.

    Returns (features [N,T,F] or None, step_errors [N,T], predictions pre-error).
    """
    n, T1 = u_ref.shape[0], u_ref.shape[1]
    T = T1 - 1
    params = _filter_solver_params(solver, params)
    feats = np.zeros((n, T, 9), dtype=np.float64) if store_features else None
    errs = np.zeros((n, T), dtype=np.float64)
    u = u_ref[:, 0].copy()
    u_prev = u.copy()
    hist = new_history(n)
    for t in range(T):
        preds = ensemble_predict(models, u, device=device)
        mean = _mean(preds)
        if store_features:
            feats[:, t] = compute_features(u_prev, u, preds, solver, dt, hist, t, T, params)
        errs[:, t] = step_error(mean, u_ref[:, t + 1])
        u_prev = u
        u = mean
    return feats, errs


def plan_scores(models, u_ref, solver, params, dt, score_kind: str, risk_model=None,
                device: str = "cpu") -> np.ndarray:
    feats, errs = neural_pass(models, u_ref, solver, params, dt, device=device)
    if score_kind == "oracle":
        return errs
    if score_kind == "residual":
        return feats[:, :, 2]
    if score_kind == "ensemble":
        return feats[:, :, 0]
    if score_kind in ("calibrated", "brace"):
        if risk_model is None:
            raise ValueError("risk model required")
        mode = "expected" if score_kind == "calibrated" else "hybrid"
        return risk_model.risk_score(feats.reshape(-1, feats.shape[-1]), mode=mode).reshape(feats.shape[:2])
    raise ValueError(score_kind)


def execute_actions(models, u_ref, actions: np.ndarray, benchmark: str, params: dict,
                    dt: float, mode: str = "step", window: int = 1,
                    device: str = "cpu", fail_tau: float = 0.1) -> dict:
    """Run the closed loop with a fixed per-(trajectory, step) action plan."""
    n, T1 = u_ref.shape[0], u_ref.shape[1]
    T = T1 - 1
    corrector = Corrector(benchmark, params, dt)
    u = u_ref[:, 0].copy()
    states = np.zeros_like(u_ref) if mode == "window" else None
    if states is not None:
        states[:, 0] = u
    errs = np.zeros((n, T))
    pre_errs = np.zeros((n, T))
    cost = np.zeros(n)
    t0 = time.time()
    for t in range(T):
        preds = ensemble_predict(models, u, device=device)
        mean = _mean(preds)
        pre_errs[:, t] = step_error(mean, u_ref[:, t + 1])
        u_next = mean
        idx = np.where(actions[:, t])[0]
        if mode == "step":
            for i in idx:
                u_next[i] = corrector.advance(u[i], i, 1)
                cost[i] += 1
        elif mode == "window":
            for i in idx:
                w = min(window, t + 1)
                u_next[i] = corrector.advance(states[i, t + 1 - w], i, w)
                cost[i] += w
        elif mode == "reanchor":
            for i in idx:
                u_next[i] = corrector.advance(u_ref[i, 0], i, t + 1)
                cost[i] += t + 1
        else:
            raise ValueError(mode)
        errs[:, t] = step_error(u_next, u_ref[:, t + 1])
        u = u_next
        if states is not None:
            states[:, t + 1] = u
    total_units = T
    return {
        "errs": errs,
        "pre_errs": pre_errs,
        "cost_fraction": float(cost.mean() / total_units),
        "cost_per_traj": cost,
        "intervention_fraction": float(actions.mean()),
        "wall_seconds": time.time() - t0,
        "metrics": summarize(errs, pre_errs, cost / total_units, actions, fail_tau),
    }


def adaptive_execute(models, u_ref, benchmark: str, params: dict, dt: float, budget: float,
                     score_kind: str, risk_model=None, mode: str = "step",
                     device: str = "cpu", fail_tau: float = 0.1) -> dict:
    """Online adaptive-threshold policy with budget pacing."""
    from .policies import AdaptiveBudgetController

    n, T1 = u_ref.shape[0], u_ref.shape[1]
    T = T1 - 1
    ctrl = AdaptiveBudgetController(n, T, budget, cost_per_action=1)
    corrector = Corrector(benchmark, params, dt)
    u = u_ref[:, 0].copy()
    u_prev = u.copy()
    hist = new_history(n)
    solver = corrector._solver(0) if n else None
    params = _filter_solver_params(solver, params) if solver is not None else params
    errs = np.zeros((n, T))
    pre_errs = np.zeros((n, T))
    actions = np.zeros((n, T), dtype=bool)
    cost = np.zeros(n)
    t_start = time.time()
    for t in range(T):
        preds = ensemble_predict(models, u, device=device)
        mean = _mean(preds)
        feats = compute_features(u_prev, u, preds, solver, dt, hist, t, T, params)
        if score_kind == "residual":
            scores = feats[:, 2]
        elif score_kind == "ensemble":
            scores = feats[:, 0]
        elif score_kind == "calibrated":
            scores = risk_model.risk_score(feats, mode="expected")
        elif score_kind == "brace":
            scores = risk_model.risk_score(feats, mode="hybrid")
        else:
            raise ValueError(score_kind)
        act = ctrl.decide(t, scores)
        pre_errs[:, t] = step_error(mean, u_ref[:, t + 1])
        u_next = mean
        for i in np.where(act)[0]:
            u_next[i] = corrector.advance(u[i], i, 1)
            cost[i] += 1
        actions[:, t] = act
        errs[:, t] = step_error(u_next, u_ref[:, t + 1])
        u_prev = u
        u = u_next
    return {
        "errs": errs,
        "pre_errs": pre_errs,
        "cost_fraction": float(cost.mean() / T),
        "cost_per_traj": cost,
        "intervention_fraction": float(actions.mean()),
        "actions": actions,
        "wall_seconds": time.time() - t_start,
        "metrics": summarize(errs, pre_errs, cost / T, actions, fail_tau),
    }


def correction_diagnostics(errs, pre_errs, actions, n_bins: int = 20) -> dict:
    """Per-correction error change: positive = correction reduced error.

    ``pre_errs`` is the error of the neural prediction at the same step, so this
    isolates the local effect of applying the reference solver.
    """
    a = np.asarray(actions, dtype=bool)
    if a.sum() == 0:
        return {"correction_gain_mean": float("nan"), "correction_harm_rate": float("nan"),
                "correction_gain_profile": [float("nan")] * n_bins,
                "correction_by_time_profile": [0.0] * n_bins}
    gain = np.where(a, pre_errs - errs, np.nan)
    T = errs.shape[1]
    edges = np.linspace(0, T, n_bins + 1).astype(int)
    gp, cp = [], []
    for i in range(n_bins):
        sl = slice(edges[i], max(edges[i] + 1, edges[i + 1]))
        g = gain[:, sl]
        gp.append(float(np.nanmean(g)) if np.any(~np.isnan(g)) else float("nan"))
        cp.append(float(np.mean(a[:, sl])))
    return {
        "correction_gain_mean": float(np.nanmean(gain)),
        "correction_harm_rate": float(np.mean(errs[a] > pre_errs[a])),
        "correction_gain_profile": gp,
        "correction_by_time_profile": cp,
    }


def summarize(errs, pre_errs, cost_fraction, actions, fail_tau=0.1) -> dict:
    per_traj_mean = errs.mean(axis=1)
    per_traj_final = errs[:, -1]
    return {
        "rel_l2_mean": float(per_traj_mean.mean()),
        "rel_l2_median": float(np.median(per_traj_mean)),
        "rel_l2_p90": float(np.quantile(per_traj_mean, 0.9)),
        "divergence_rate": float(np.mean(per_traj_mean > 1.0)),
        "rel_l2_std": float(per_traj_mean.std(ddof=1)) if len(per_traj_mean) > 1 else 0.0,
        "rel_l2_final": float(per_traj_final.mean()),
        "rel_l2_max": float(errs.max(axis=1).mean()),
        "step_failure_rate": float((errs > fail_tau).mean()),
        "traj_failure_rate": float((errs.max(axis=1) > fail_tau).mean()),
        "pre_rel_l2_mean": float(pre_errs.mean()),
        "pre_failure_rate": float((pre_errs > fail_tau).mean()),
        "cost_fraction": float(np.mean(cost_fraction)),
        "intervention_fraction": float(actions.mean()),
        "per_traj_mean": per_traj_mean.tolist(),
        "per_traj_final": per_traj_final.tolist(),
        "per_step_mean": errs.mean(axis=0).tolist(),
        "action_profile": actions.mean(axis=0).tolist(),
        "action_time_mean": float((actions * np.arange(actions.shape[1])[None, :]).sum()
                                  / max(actions.sum(), 1)),
        **correction_diagnostics(errs, pre_errs, actions),
    }
