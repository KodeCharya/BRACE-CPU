"""Phase 10 ablation and robustness experiments.

Reuses existing surrogate checkpoints (seed 0-2) to save time. Runs:
  A1. Feature ablation: each single risk feature vs combined (AUROC on test)
  A2. Calibration-set size sweep (train risk model on fraction of cal set)
  A3. OOD robustness: evaluate surrogate + risk on ood_test partition

All on CPU, sequential, small batches.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict

import numpy as np

from models.surrogate import NeuralSurrogate
from models.trainer import collect_npz_paths
from uncertainty.risk_estimator import RiskEstimator
from uncertainty.risk_model import LogisticRiskModel
from uncertainty.calibrator import PlattScaling, expected_calibration_error, brier_score, failure_metrics
from uncertainty.features import extract_features, feature_names
from solvers.base import relative_l2_error

TAU = 0.01
L = 1.0


def load_surrogate(seed: int) -> NeuralSurrogate:
    s = NeuralSurrogate()
    s.load(f"checkpoints/surrogate_seed{seed}.pt", device="cpu")
    return s


def extract_pairs(surrogate, paths, stride: int = 4):
    """Return (X, y, errs) feature matrix and failure labels from trajectories."""
    X, y, errs = [], [], []
    from uncertainty.features import burgers_pde_residual
    for path in paths:
        data = np.load(path)
        states = data["states"].astype(np.float64)
        times = data["times"].astype(np.float64)
        data.close()
        meta = json.load(open(path[:-4] + ".json"))
        nu = float(meta.get("nu", 0.01))
        params = np.array([nu], dtype=np.float64)
        dt = float(times[1] - times[0]) if times.size > 1 else 0.001
        for i in range(0, states.shape[0] - 1, stride):
            u_t = states[i]; u_true = states[i + 1]; t = float(times[i])
            u_pred = surrogate.predict(u_t, params, np.array([t]))
            feats = extract_features(surrogate, u_t, u_pred, params, t, dt, L)
            err = relative_l2_error(u_pred, u_true)
            X.append(feats); y.append(1.0 if err > TAU else 0.0); errs.append(err)
    return np.stack(X), np.array(y), np.array(errs)


def sigmoid(z):
    z = np.clip(z, -60, 60)
    return 1.0 / (1.0 + np.exp(-z))


def auroc(scores, y):
    pos = scores[y == 1]; neg = scores[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    greater = np.sum(pos[:, None] > neg[None, :])
    equal = np.sum(pos[:, None] == neg[None, :])
    return float((greater + 0.5 * equal) / (len(pos) * len(neg)))


def fit_logistic_on_features(X, y, seed=0, lr=0.1, epochs=3000, l2=1e-3):
    m = LogisticRiskModel(lr=lr, epochs=epochs, l2=l2, seed=seed)
    m.fit(X, y)
    return m


# ---------------------------------------------------------------------------
# A1. Feature ablation
# ---------------------------------------------------------------------------

def feature_ablation(seed: int, test_paths):
    print(f"\n=== A1 Feature ablation (seed {seed}) ===")
    surrogate = load_surrogate(seed)
    X, y, errs = extract_pairs(surrogate, test_paths, stride=4)
    names = feature_names()
    n_feat = X.shape[1]

    # Combined
    m_comb = fit_logistic_on_features(X, y, seed=seed)
    p_comb = sigmoid(m_comb.predict_logit(X))
    auc_comb = auroc(p_comb, y)
    print(f"  combined (all {n_feat} feats): AUROC={auc_comb:.4f}")

    # Single features
    for j in range(n_feat):
        m = fit_logistic_on_features(X[:, j:j + 1], y, seed=seed)
        p = sigmoid(m.predict_logit(X[:, j:j + 1]))
        a = auroc(p, y)
        print(f"  {names[j]:<28} AUROC={a:.4f}")

    # Leave-one-out
    for j in range(n_feat):
        mask = [k for k in range(n_feat) if k != j]
        m = fit_logistic_on_features(X[:, mask], y, seed=seed)
        p = sigmoid(m.predict_logit(X[:, mask]))
        a = auroc(p, y)
        print(f"  w/o {names[j]:<24} AUROC={a:.4f}")


# ---------------------------------------------------------------------------
# A2. Calibration-set size sweep
# ---------------------------------------------------------------------------

def calset_size_sweep(seed: int, cal_paths, test_paths, fractions=(0.25, 0.5, 1.0)):
    print(f"\n=== A2 Calibration-set size sweep (seed {seed}) ===")
    surrogate = load_surrogate(seed)
    # Fit risk model on FULL cal set once (fixed), then vary Platt cal-set size
    X_cal, y_cal, _ = extract_pairs(surrogate, cal_paths, stride=4)
    m_risk = fit_logistic_on_features(X_cal, y_cal, seed=seed)
    raw_cal = m_risk.predict_logit(X_cal)

    X_test, y_test, _ = extract_pairs(surrogate, test_paths, stride=4)
    raw_test = m_risk.predict_logit(X_test)

    n_cal = len(raw_cal)
    for frac in fractions:
        n = max(10, int(n_cal * frac))
        idx = np.random.default_rng(seed).choice(n_cal, size=n, replace=False)
        ps = PlattScaling(lr=0.2, epochs=5000, seed=seed)
        ps.fit(raw_cal[idx], y_cal[idx])
        p_test = ps.calibrate(raw_test)
        ece = expected_calibration_error(p_test, y_test)
        brier = brier_score(p_test, y_test)
        m = failure_metrics(p_test, y_test, 0.5)
        print(f"  cal fraction={frac:.2f} (n={n}): ECE={ece:.4f} Brier={brier:.4f} "
              f"AUROC={m['auroc']:.4f} P={m['precision']:.3f} R={m['recall']:.3f}")


# ---------------------------------------------------------------------------
# A3. OOD robustness
# ---------------------------------------------------------------------------

def ood_robustness(seed: int, ood_paths):
    print(f"\n=== A3 OOD robustness (seed {seed}) ===")
    surrogate = load_surrogate(seed)
    X, y, errs = extract_pairs(surrogate, ood_paths, stride=4)
    m = fit_logistic_on_features(X, y, seed=seed)
    p = sigmoid(m.predict_logit(X))
    auc = auroc(p, y)
    fail_rate = float(y.mean())
    mean_err = float(errs.mean())
    print(f"  OOD: n={len(X)} fail_rate={fail_rate:.3f} mean_err={mean_err:.5f} AUROC={auc:.4f}")


def main():
    test_paths = collect_npz_paths("data/processed/burgers/test")
    cal_paths = collect_npz_paths("data/processed/burgers/calibration")
    ood_paths = collect_npz_paths("data/processed/burgers/ood_test")

    for seed in [0, 1, 2]:
        feature_ablation(seed, test_paths)
        calset_size_sweep(seed, cal_paths, test_paths)
        ood_robustness(seed, ood_paths)


if __name__ == "__main__":
    main()
