"""Lightweight failure-risk model: regularized logistic regression (numpy).

Maps a risk feature vector to P(failure) where failure = next-step relative
L2 error exceeding a tolerance tau. Implemented in pure NumPy to avoid heavy
dependencies and keep the CPU footprint small.
"""

from __future__ import annotations

import numpy as np


class LogisticRiskModel:
    """L2-regularized logistic regression trained with gradient descent."""

    def __init__(self, lr: float = 0.1, epochs: int = 3000, l2: float = 1e-3,
                 seed: int = 0, verbose: bool = False):
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.seed = seed
        self.verbose = verbose
        self.w: np.ndarray | None = None
        self.b: float = 0.0
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def _standardize_fit(self, X: np.ndarray) -> np.ndarray:
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + 1e-8
        return (X - self.mean_) / self.std_

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRiskModel":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        Xs = self._standardize_fit(X)
        n, d = Xs.shape
        rng = np.random.default_rng(self.seed)
        self.w = rng.standard_normal(d) * 0.1
        self.b = 0.0
        for epoch in range(self.epochs):
            z = Xs @ self.w + self.b
            p = self._sigmoid(z)
            # gradient of BCE + L2
            grad_w = (Xs.T @ (p - y)) / n + self.l2 * self.w
            grad_b = float(np.mean(p - y))
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
            if self.verbose and epoch % max(1, self.epochs // 5) == 0:
                loss = float(-np.mean(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12)))
                print(f"  risk epoch {epoch}: BCE={loss:.4f}")
        return self

    def predict_logit(self, X: np.ndarray) -> np.ndarray:
        assert self.w is not None
        Xs = self._standardize(np.asarray(X, dtype=np.float64))
        return Xs @ self.w + self.b

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._sigmoid(self.predict_logit(X))

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))

