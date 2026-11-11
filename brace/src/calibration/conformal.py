"""Split conformal calibration of surrogate error estimates.

Guarantee (distribution-free, exchangeable calibration/test data):
the interval [exp(log r_hat - q), exp(log r_hat + q)] covers the true
one-step relative error with probability >= 1 - alpha. With alpha = 0.1 the
one-sided upper bound exp(log r_hat + q) has coverage >= 0.95.
Exchangeability fails under distribution shift; we measure empirical
coverage on OOD splits instead of assuming it.
"""
from __future__ import annotations

import numpy as np


def conformal_quantile(scores: np.ndarray, alpha: float = 0.1) -> float:
    scores = np.sort(np.asarray(scores, dtype=np.float64))
    n = len(scores)
    if n == 0:
        return float("inf")
    k = int(np.ceil((n + 1) * (1 - alpha)))
    k = min(max(k, 1), n)
    return float(scores[k - 1])


class ConformalError:
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.q = float("inf")

    def fit(self, log_error_pred: np.ndarray, errors: np.ndarray) -> "ConformalError":
        log_true = np.log(np.maximum(errors, 1e-8))
        self.q = conformal_quantile(np.abs(log_true - log_error_pred), self.alpha)
        return self

    def upper(self, log_error_pred: np.ndarray) -> np.ndarray:
        return np.exp(log_error_pred + self.q)

    def lower(self, log_error_pred: np.ndarray) -> np.ndarray:
        return np.exp(log_error_pred - self.q)

    def coverage(self, log_error_pred: np.ndarray, errors: np.ndarray) -> float:
        lo = self.lower(log_error_pred)
        hi = self.upper(log_error_pred)
        return float(np.mean((errors >= lo) & (errors <= hi)))

    def upper_coverage(self, log_error_pred: np.ndarray, errors: np.ndarray) -> float:
        return float(np.mean(errors <= self.upper(log_error_pred)))
