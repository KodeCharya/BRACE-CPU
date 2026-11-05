"""Calibration of raw risk scores into well-calibrated failure probabilities.

Provides Platt scaling (a 1D logistic regression mapping raw score -> calibrated
probability) and reliability-diagram statistics (calibration error, Brier score).

The calibration map is fit on a held-out calibration portion, never on test data.
"""

from __future__ import annotations

import numpy as np

from .risk_model import LogisticRiskModel


class PlattScaling:
    """Platt scaling: P(failure | score) = sigmoid(a * score + b)."""

    def __init__(self, lr: float = 0.2, epochs: int=5000, l2: float = 1e-3, seed: int = 0):
        self.model = LogisticRiskModel(lr=lr, epochs=epochs, l2=l2, seed=seed)
        self.fitted: bool = False

    def fit(self, scores: np.ndarray, y: np.ndarray) -> "PlattScaling":
        scores = np.asarray(scores, dtype=np.float64).reshape(-1, 1)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        self.model.fit(scores, y)
        self.fitted = True
        return self

    def calibrate(self, scores: np.ndarray) -> np.ndarray:
        assert self.fitted, "Platt scaling not fitted"
        scores = np.asarray(scores, dtype=np.float64).reshape(-1, 1)
        return self.model.predict_proba(scores).reshape(-1)


def brier_score(probs: np.ndarray, y: np.ndarray) -> float:
    probs = np.asarray(probs, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return float(np.mean((probs - y) ** 2))


def expected_calibration_error(probs: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (ECE) with equal-width bins in [0,1]."""
    probs = np.asarray(probs, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = len(probs)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (probs >= lo) & (probs < hi) if i < n_bins - 1 else (probs >= lo) & (probs <= hi)
        if not np.any(mask):
            continue
        conf = float(np.mean(probs[mask]))
        acc = float(np.mean(y[mask]))
        ece += (mask.sum() / total) * abs(conf - acc)
    return ece


def reliability_table(probs: np.ndarray, y: np.ndarray, n_bins: int = 10):
    """Return per-bin (bin_idx, conf, acc, count) for a reliability diagram."""
    probs = np.asarray(probs, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (probs >= lo) & (probs < hi) if i < n_bins - 1 else (probs >= lo) & (probs <= hi)
        count = int(mask.sum())
        if count == 0:
            rows.append((i, None, None, 0))
            continue
        conf = float(np.mean(probs[mask]))
        acc = float(np.mean(y[mask]))
        rows.append((i, conf, acc, count))
    return rows


def failure_metrics(probs: np.ndarray, y: np.ndarray, threshold: float = 0.5):
    """Precision, recall, F1, AUROC-ish (via rank) for failure detection at a threshold."""
    probs = np.asarray(probs, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    pred = (probs >= threshold).astype(int)
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    # AUROC via Mann-Whitney U
    auroc = _auroc(probs, y)
    return {
        "precision": precision, "recall": recall, "f1": f1,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn, "auroc": auroc,
    }


def _auroc(scores: np.ndarray, y: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    pos = scores[y == 1]
    neg = scores[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # Mann-Whitney U
    greater = np.sum(pos[:, None] > neg[None, :])
    equal = np.sum(pos[:, None] == neg[None, :])
    return float((greater + 0.5 * equal) / (len(pos) * len(neg)))
