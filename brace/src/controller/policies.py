"""Risk scores and budget allocation policies."""
from __future__ import annotations

import numpy as np

from ..uncertainty.risk import RiskModel


def score_from_features(kind: str, feats: np.ndarray, risk_model: RiskModel | None,
                        preds: np.ndarray | None = None, true_error: np.ndarray | None = None) -> np.ndarray:
    if kind == "residual":
        return feats[:, 2]
    if kind == "ensemble":
        return feats[:, 0]
    if kind == "calibrated":
        if risk_model is None:
            raise ValueError("calibrated score needs a fitted risk model")
        return risk_model.risk_score(feats, mode="expected")
    if kind == "brace":
        if risk_model is None:
            raise ValueError("brace score needs a fitted risk model")
        return risk_model.risk_score(feats, mode="hybrid")
    if kind == "oracle":
        if true_error is None:
            raise ValueError("oracle score needs the true error")
        return true_error
    raise ValueError(kind)


def topk_actions(scores: np.ndarray, budget_fraction: float, cost_per_action: int = 1,
                 min_spacing: int = 1) -> np.ndarray:
    """Per-trajectory top-k selection with an exact budget cap.

    k = floor(B * T / cost_per_action). ``min_spacing`` prevents overlapping
    correction windows (spacing 1 = no constraint).
    """
    n, T = scores.shape
    k = int(np.floor(budget_fraction * T / cost_per_action))
    k = max(0, min(k, T))
    actions = np.zeros((n, T), dtype=bool)
    if k == 0:
        return actions
    order = np.argsort(-scores, axis=1)
    for i in range(n):
        chosen = 0
        taken = np.zeros(T, dtype=bool)
        for t in order[i]:
            if taken[t]:
                continue
            if min_spacing > 1 and np.any(taken[max(0, t - min_spacing + 1):t + min_spacing]):
                continue
            taken[t] = True
            actions[i, t] = True
            chosen += 1
            if chosen >= k:
                break
    return actions


def quantile_threshold(scores: np.ndarray, budget_fraction: float) -> float:
    """Static score threshold calibrated to intervene on ~B of steps."""
    scores = np.asarray(scores, dtype=np.float64).ravel()
    if len(scores) == 0:
        return float("inf")
    q = 1.0 - budget_fraction
    return float(np.quantile(scores, q))


class AdaptiveBudgetController:
    """Online quantile threshold with cumulative budget pacing."""

    def __init__(self, n_traj: int, horizon: int, budget: float, cost_per_action: int = 1,
                 window: int = 50):
        self.B = budget
        self.T = horizon
        self.cost = cost_per_action
        self.window = window
        self.buf = [[] for _ in range(n_traj)]
        self.spent = np.zeros(n_traj, dtype=np.int64)  # in cost units
        self.total_units = horizon * 1  # 1 cost unit == one full interval (normalised)

    def decide(self, t: int, scores: np.ndarray) -> np.ndarray:
        n = len(scores)
        out = np.zeros(n, dtype=bool)
        for i in range(n):
            b = self.buf[i]
            b.append(float(scores[i]))
            if len(b) > self.window:
                b.pop(0)
            if len(b) < 10:
                continue
            tau = np.quantile(b, 1.0 - self.B)
            allowed_frac = self.B * (t + 1) / self.T + 0.05 * self.B
            if self.spent[i] + self.cost <= allowed_frac * self.T and scores[i] >= tau:
                out[i] = True
                self.spent[i] += self.cost
        return out


def threshold_actions(scores: np.ndarray, tau: float) -> np.ndarray:
    return np.asarray(scores) >= tau
