"""Fit the calibrated risk model on validation trajectories."""
from __future__ import annotations

import numpy as np

from ..calibration.conformal import ConformalError
from ..controller.rollout import neural_pass
from ..correction.numerical import make_solver
from ..uncertainty.risk import RiskModel, auroc, expected_calibration_error
from ..uncertainty.features import FEATURE_NAMES
from scipy import stats


def discounted_future(errs: np.ndarray, gamma: float = 0.95, tau: float = 0.05):
    """Discounted future error sum and any-future-failure labels from [N,T] errors."""
    n_t = errs.shape[1]
    fut = np.zeros_like(errs)
    fut[:, -1] = errs[:, -1]
    for t in range(n_t - 2, -1, -1):
        fut[:, t] = errs[:, t] + gamma * fut[:, t + 1]
    labels = (errs > tau).astype(np.float64)
    any_fail = np.zeros_like(labels)
    any_fail[:, -1] = labels[:, -1]
    for t in range(n_t - 2, -1, -1):
        any_fail[:, t] = np.maximum(labels[:, t], any_fail[:, t + 1])
    return fut, any_fail


def filter_params(benchmark: str, params: dict) -> dict:
    keys = {"burgers1d": ("nu",), "allencahn1d": ("eps",),
            "advection2d": ("nu", "cx", "cy")}[benchmark]
    return {k: params[k] for k in keys if k in params}


def residual_solver(benchmark: str, dt_save: float, n: int | None = None):
    """Solver used only for spatial operators in residual features."""
    if benchmark == "burgers1d":
        return make_solver(benchmark, {"nu": 0.01}, dt_save, n=n)[0]
    if benchmark == "allencahn1d":
        return make_solver(benchmark, {"eps": 0.01}, dt_save, n=n)[0]
    return make_solver(benchmark, {"nu": 0.01, "cx": 1.0, "cy": 0.5}, dt_save, n=n)[0]


def fit_risk_model(models, u_val: np.ndarray, benchmark: str, params: dict, dt: float,
                   tau_fail: float = 0.05, alpha: float = 0.1,
                   cal_frac: float = 0.5, mask=None, risk_target: str = "step",
                   gamma: float = 0.95) -> dict:
    solver = residual_solver(benchmark, dt, n=u_val.shape[-1])
    params = filter_params(benchmark, params)
    feats, errs = neural_pass(models, u_val, solver, params, dt)
    n = feats.shape[0]
    n_fit = max(2, int(round(n * (1 - cal_frac))))
    X = feats.reshape(-1, feats.shape[-1])
    e = errs.reshape(-1)
    X_fit, e_fit = feats[:n_fit].reshape(-1, feats.shape[-1]), errs[:n_fit].reshape(-1)
    X_cal, e_cal = feats[n_fit:].reshape(-1, feats.shape[-1]), errs[n_fit:].reshape(-1)

    if risk_target == "horizon":
        fut_fit, lab_fit = discounted_future(errs[:n_fit], gamma)
        fut_cal, lab_cal = discounted_future(errs[n_fit:], gamma)
        log_fit, log_cal = np.log(np.maximum(fut_fit, 1e-6)), np.log(np.maximum(fut_cal, 1e-6))
        risk = RiskModel(tau_fail=tau_fail, mask=mask).fit(
            X_fit, e_fit, labels=lab_fit.reshape(-1), log_target=log_fit.reshape(-1))
        risk.horizon_tau = float(np.quantile(fut_fit, 0.75))
        conf = ConformalError(alpha=alpha).fit(
            risk.predict_log_error(X_cal), np.exp(log_cal.reshape(-1)))
    else:
        risk = RiskModel(tau_fail=tau_fail, mask=mask).fit(X_fit, e_fit)
        conf = ConformalError(alpha=alpha).fit(risk.predict_log_error(X_cal), e_cal)
    report = risk.report(X, e)
    report.update({
        "conformal_q": conf.q,
        "conformal_coverage_val": conf.coverage(risk.predict_log_error(X_cal), e_cal),
        "conformal_upper_coverage_val": conf.upper_coverage(risk.predict_log_error(X_cal), e_cal),
        "n_val_features": int(n),
        "n_fit": int(n_fit),
        "failure_rate_val": float((e > tau_fail).mean()),
    })
    # Validation score distributions used to set static thresholds.
    scores = {
        "residual": feats[:, :, 2],
        "ensemble": feats[:, :, 0],
        "calibrated": risk.risk_score(X, mode="expected").reshape(errs.shape),
        "brace": risk.risk_score(X, mode="hybrid").reshape(errs.shape),
    }
    return {"risk_model": risk, "conformal": conf, "report": report,
            "features": feats, "errors": errs, "scores": scores}


def evaluate_risk_on_split(models, u: np.ndarray, benchmark: str, params: dict, dt: float,
                           risk: RiskModel, conf: ConformalError, tau_fail: float = 0.05,
                           risk_target: str = "step", gamma: float = 0.95,
                           risk_hz: RiskModel | None = None, conf_hz: ConformalError | None = None) -> dict:
    solver = residual_solver(benchmark, dt, n=u.shape[-1])
    params = filter_params(benchmark, params)
    feats, errs = neural_pass(models, u, solver, params, dt)
    X = feats.reshape(-1, feats.shape[-1])
    e = errs.reshape(-1)
    p = risk.predict_proba(X)
    y = (e > tau_fail).astype(float)
    log_pred = risk.predict_log_error(X)
    out = risk.report(X, e)
    fut, _ = discounted_future(errs, gamma, tau_fail)
    log_fut = np.log(np.maximum(fut.reshape(-1), 1e-8))
    if risk_hz is not None:
        pred_fut = risk_hz.predict_log_error(X)
        rho = stats.spearmanr(pred_fut, log_fut).statistic
        out["horizon_spearman"] = float(rho)
        out["horizon_conformal_coverage"] = (float(conf_hz.coverage(pred_fut, np.exp(log_fut)))
                                             if conf_hz else float("nan"))
        out["horizon_pred_mean"] = float(np.mean(np.exp(pred_fut)))
        out["horizon_true_mean"] = float(np.mean(np.exp(log_fut)))
    out.update({
        "conformal_coverage": conf.coverage(log_pred, e),
        "conformal_upper_coverage": conf.upper_coverage(log_pred, e),
        "auroc_score_expected": auroc(risk.predict_error(X), y),
        "risk_target": risk_target,
        "auroc_residual": auroc(feats[:, :, 2].ravel(), y),
        "auroc_ensemble": auroc(feats[:, :, 0].ravel(), y),
        "auroc_calibrated": auroc(p, y),
        "auroc_brace": auroc(risk.risk_score(X, mode="hybrid"), y),
        "auroc_oracle_target": 1.0,
        "ece": expected_calibration_error(p, y),
        "brier": float(np.mean((p - y) ** 2)),
        "failure_rate": float(y.mean()),
        "risk_coverage": risk_coverage(risk.risk_score(X, mode="hybrid"), e),
        "risk_coverage_residual": risk_coverage(feats[:, :, 2].ravel(), e),
        "risk_coverage_ensemble": risk_coverage(feats[:, :, 0].ravel(), e),
        "reliability": reliability_curve(p, y),
        "auroc_features": {name: auroc(feats[:, :, j].ravel(), y)
                           for j, name in enumerate(FEATURE_NAMES)},
    })
    return out


def reliability_curve(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> dict:
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if m.sum() > 0:
            rows.append({"bin_center": float((bins[b] + bins[b + 1]) / 2),
                         "mean_pred": float(p[m].mean()),
                         "observed": float(y[m].mean()),
                         "count": int(m.sum())})
    return {"bins": rows}


def risk_coverage(scores: np.ndarray, errors: np.ndarray, n_points: int = 21) -> dict:
    """Selective-prediction curve: mean error of accepted samples vs coverage.

    Low scores are treated as confident (accepted first). A useful score
    yields a monotonically increasing curve.
    """
    order = np.argsort(scores)
    err = np.asarray(errors)[order]
    n = len(err)
    coverages = np.linspace(0.05, 1.0, n_points)
    means = [float(err[: max(1, int(c * n))].mean()) for c in coverages]
    return {"coverage": coverages.tolist(), "mean_error": means}
