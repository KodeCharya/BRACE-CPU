"""Calibrated risk model: failure probability + expected error magnitude.

The model is fit ONLY on validation trajectories (never test). It combines
standardised online features into
  * p_fail : logistic probability that the one-step relative error > tau
  * log_err: ridge estimate of log(one-step relative error)

Split conformal calibration of the error estimate is provided in
``src/calibration/conformal.py``.
"""
from __future__ import annotations

import numpy as np

from ..uncertainty.features import FEATURE_NAMES


def _standardize_fit(X: np.ndarray):
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-8
    return mu, sd


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUROC (ties handled by average ranks)."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(bool)
    n_pos, n_neg = labels.sum(), (~labels).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ties
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        sums = np.bincount(inv, weights=ranks)
        ranks = (sums / counts)[inv]
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def expected_calibration_error(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() == 0:
            continue
        ece += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return float(ece)


class RiskModel:
    def __init__(self, tau_fail: float = 0.05, l2: float = 1e-3, mask=None):
        self.tau_fail = tau_fail
        self.l2 = l2
        self.mask = None if mask is None else np.asarray(mask, dtype=bool)
        self.feature_names = [n for n, keep in zip(FEATURE_NAMES, self.mask)] if self.mask is not None else FEATURE_NAMES
        self.horizon_tau = None

    def _apply(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return X[:, self.mask] if self.mask is not None else X
        self.w_logit = None
        self.b_logit = 0.0
        self.w_ridge = None
        self.b_ridge = 0.0
        self.mu = None
        self.sd = None

    def fit(self, X: np.ndarray, errors: np.ndarray, labels: np.ndarray | None = None,
            log_target: np.ndarray | None = None) -> "RiskModel":
        X = self._apply(X)
        y = (errors > self.tau_fail).astype(np.float64) if labels is None else np.asarray(labels, dtype=np.float64)
        self.mu, self.sd = _standardize_fit(X)
        Z = (X - self.mu) / self.sd
        n, d = Z.shape
        # Logistic regression, full-batch Newton / IRLS with ridge.
        w = np.zeros(d)
        b = np.log(max(y.mean(), 1e-3) / max(1 - y.mean(), 1e-3))
        for _ in range(100):
            p = _sigmoid(Z @ w + b)
            W = p * (1 - p) + 1e-9
            grad = Z.T @ (p - y) + self.l2 * n * w
            H = (Z * W[:, None]).T @ Z + self.l2 * n * np.eye(d)
            step = np.linalg.solve(H, grad)
            w -= step
            b -= np.mean(p - y)
            if np.max(np.abs(step)) < 1e-8:
                break
        self.w_logit, self.b_logit = w, b
        # Ridge regression on log error (or a supplied horizon target).
        target = (np.log(np.maximum(errors, 1e-6)) if log_target is None
                  else np.asarray(log_target, dtype=np.float64))
        A = Z.T @ Z + self.l2 * n * np.eye(d)
        self.w_ridge = np.linalg.solve(A, Z.T @ (target - target.mean()))
        self.b_ridge = float(target.mean())
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Z = (self._apply(X) - self.mu) / self.sd
        return _sigmoid(Z @ self.w_logit + self.b_logit)

    def predict_log_error(self, X: np.ndarray) -> np.ndarray:
        Z = (self._apply(X) - self.mu) / self.sd
        return Z @ self.w_ridge + self.b_ridge

    def predict_error(self, X: np.ndarray) -> np.ndarray:
        return np.exp(self.predict_log_error(X))

    def risk_score(self, X: np.ndarray, mode: str = "expected") -> np.ndarray:
        """Score used by the budget allocation controller."""
        if mode == "prob":
            return self.predict_proba(X)
        if mode == "expected":
            return self.predict_error(X)
        if mode == "hybrid":
            return self.predict_proba(X) * self.predict_error(X)
        raise ValueError(mode)

    def report(self, X: np.ndarray, errors: np.ndarray) -> dict:
        p = self.predict_proba(X)
        y = (errors > self.tau_fail).astype(float)
        return {
            "auroc": auroc(p, y),
            "auroc_expected": auroc(self.predict_error(X), y),
            "brier": float(np.mean((p - y) ** 2)),
            "ece": expected_calibration_error(p, y),
            "base_rate": float(y.mean()),
            "n": int(len(y)),
            "tau_fail": self.tau_fail,
            "feature_names": self.feature_names,
            "logit_weights": dict(zip(self.feature_names, self.w_logit.tolist())),
        }
