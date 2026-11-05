"""Failure-risk estimator: features + risk model + calibration.

Failure is defined a priori as next-step relative L2 error exceeding a
tolerance tau. The estimator is fit on a separate calibration dataset (never
test labels). The calibration set is split into a risk-model training half and
a calibration-map (Platt scaling) half to avoid double-dipping.
"""

from __future__ import annotations

import json
import os
from typing import List, Tuple

import numpy as np

from .features import extract_features, feature_names
from .risk_model import LogisticRiskModel
from .calibrator import PlattScaling, brier_score, expected_calibration_error, failure_metrics

from solvers.base import relative_l2_error


class RiskEstimator:
    def __init__(self, tau: float = 0.01, L: float = 1.0,
                 risk_model_kwargs: dict | None = None,
                 platt_kwargs: dict | None = None,
                 seed: int = 0):
        """tau: failure tolerance (next-step relative L2 error), set a priori."""
        self.tau = float(tau)
        self.L = float(L)
        self.risk_model_kwargs = risk_model_kwargs or {}
        self.platt_kwargs = platt_kwargs or {}
        self.seed = int(seed)
        self.risk_model: LogisticRiskModel | None = None
        self.platt: PlattScaling | None = None
        self.fitted = False

    # ------------------------------------------------------------------ #
    # Fit on calibration trajectories (npz paths)
    # ------------------------------------------------------------------ #
    def fit(self, surrogate, cal_npz_paths: List[str], stride: int = 1) -> dict:
        """Extract features/labels from calibration trajectories, fit model +
        Platt scaling. Returns fit diagnostics."""
        X_list, y_list, errors = [], [], []
        for path in cal_npz_paths:
            data = np.load(path)
            states = data["states"].astype(np.float64)
            times = data["times"].astype(np.float64)
            data.close()
            meta = json.load(open(path[:-4] + ".json"))
            nu = float(meta.get("nu", 0.01))
            params = np.array([nu], dtype=np.float64)
            dt = float(times[1] - times[0]) if times.size > 1 else 0.001
            for i in range(0, states.shape[0] - 1, stride):
                u_t = states[i]
                u_true = states[i + 1]
                t = float(times[i])
                u_pred = surrogate.predict(u_t, params, np.array([t]))
                feats = extract_features(surrogate, u_t, u_pred, params, t, dt, self.L)
                err = relative_l2_error(u_pred, u_true)
                X_list.append(feats)
                y_list.append(1.0 if err > self.tau else 0.0)
                errors.append(err)
        X = np.stack(X_list, axis=0)
        y = np.stack(y_list, axis=0)
        errors = np.array(errors)

        # Split calibration: risk-model train half / Platt map half
        n = len(X)
        perm = np.random.default_rng(self.seed).permutation(n)
        n_risk = max(1, n // 2)
        risk_idx = perm[:n_risk]
        platt_idx = perm[n_risk:]

        self.risk_model = LogisticRiskModel(**self.risk_model_kwargs)
        self.risk_model.fit(X[risk_idx], y[risk_idx])

        raw_scores = self.risk_model.predict_logit(X[platt_idx])  # logits
        self.platt = PlattScaling(**self.platt_kwargs)
        self.platt.fit(raw_scores, y[platt_idx])

        # Diagnostics on the Platt half (calibration-set reliability)
        cal_probs = self.platt.calibrate(raw_scores)
        diag = {
            "n_calibration": n,
            "n_risk_train": int(n_risk),
            "n_platt": int(len(platt_idx)),
            "failure_rate": float(np.mean(y)),
            "brier": brier_score(cal_probs, y[platt_idx]),
            "ece": expected_calibration_error(cal_probs, y[platt_idx]),
            "tau": self.tau,
        }
        # failure metrics at default 0.5 threshold on the Platt half
        diag.update({f"cal_{k}": v for k, v in failure_metrics(cal_probs, y[platt_idx]).items()})
        self.fitted = True
        return diag

    # ------------------------------------------------------------------ #
    # Score
    # ------------------------------------------------------------------ #
    def score(self, surrogate, u_t: np.ndarray, params: np.ndarray, t: float, dt: float) -> float:
        """Calibrated P(failure) in [0,1] for a single step."""
        assert self.fitted, "RiskEstimator not fitted"
        u_pred = surrogate.predict(u_t, params, np.array([t]))
        feats = extract_features(surrogate, u_t, u_pred, params, t, dt, self.L)
        logit = self.risk_model.predict_logit(feats[None, :])[0]
        return float(self.platt.calibrate(np.array([logit]))[0])

    def predict_failure(self, surrogate, u_t: np.ndarray, params: np.ndarray,
                        t: float, dt: float, threshold: float = 0.5) -> bool:
        return self.score(surrogate, u_t, params, t, dt) >= threshold


if __name__ == "__main__":
    print("RiskEstimator module ready.")
