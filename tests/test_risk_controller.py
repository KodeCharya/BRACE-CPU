"""Tests for risk estimation, calibration, and the budget-aware controller.

Run from project root:
    PYTHONPATH=src .venv/bin/python -m pytest tests/ -v
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from uncertainty.risk_model import LogisticRiskModel, accuracy
from uncertainty.calibrator import (
    PlattScaling, brier_score, expected_calibration_error, failure_metrics,
)
from uncertainty.risk_estimator import RiskEstimator
from uncertainty.features import extract_features, burgers_pde_residual

from models.surrogate import NeuralSurrogate, SurrogateConfig
from models.trainer import collect_npz_paths

from controller.policies import Policy, ControlContext, Controller
from controller.rollout import run_policy
from solvers.burgers import BurgersSolver


# ---------------------------------------------------------------------------
# Risk model
# ---------------------------------------------------------------------------

def test_logistic_risk_model_separates_classes():
    rng = np.random.default_rng(0)
    X = np.concatenate([rng.normal(0, 1, (200, 2)), rng.normal(3, 1, (200, 2))])
    y = np.array([0] * 200 + [1] * 200)
    m = LogisticRiskModel(lr=0.2, epochs=4000, l2=1e-3, seed=0)
    m.fit(X, y)
    p0 = m.predict_proba(X[y == 0])
    p1 = m.predict_proba(X[y == 1])
    assert p1.mean() > p0.mean() + 0.2
    assert accuracy(y, m.predict(X)) > 0.7


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def test_platt_scaling_maps_to_unit_interval():
    ps = PlattScaling(lr=0.5, epochs=3000, seed=0)
    scores = np.linspace(-3, 3, 50)
    y = (scores > 0).astype(float)
    ps.fit(scores, y)
    out = ps.calibrate(scores)
    assert np.all(out >= 0.0) and np.all(out <= 1.0)


def test_brier_and_ece_bounds():
    rng = np.random.default_rng(1)
    probs = rng.random(100)
    y = rng.integers(0, 2, 100).astype(float)
    b = brier_score(probs, y)
    e = expected_calibration_error(probs, y)
    assert 0.0 <= b <= 1.0
    assert 0.0 <= e <= 1.0


def test_failure_metrics_keys():
    probs = np.array([0.1, 0.4, 0.6, 0.9])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    m = failure_metrics(probs, y, threshold=0.5)
    for key in ["precision", "recall", "f1", "auroc", "tp", "fp", "fn", "tn"]:
        assert key in m


# ---------------------------------------------------------------------------
# No future-ground-truth leakage in risk features
# ---------------------------------------------------------------------------

def test_risk_features_do_not_use_ground_truth():
    # extract_features must not take u_true as an argument; it only sees
    # u_t, u_pred (model's own prediction), params, t, dt.
    import inspect
    sig = inspect.signature(extract_features)
    names = list(sig.parameters.keys())
    assert "u_true" not in names
    assert "target" not in names
    assert "ground_truth" not in names
    # residual uses u_t and u_pred (prediction), not the true next state
    u_t = np.sin(np.linspace(0, 2 * np.pi, 32))
    u_pred = u_t + 0.01 * np.cos(np.linspace(0, 2 * np.pi, 32))
    res = burgers_pde_residual(u_t, u_pred, 0.01, 0.001, 1.0)
    assert res.shape == u_t.shape


# ---------------------------------------------------------------------------
# Controller: budget enforcement for policy G
# ---------------------------------------------------------------------------

class _MockSurrogate:
    def predict(self, u, params, t):
        return u.copy()


class _MockSolver:
    def advance(self, u, t, dt, n_steps=1):
        return u.copy()


class _MockRiskEstimator:
    def __init__(self, risks):
        self.risks = np.asarray(risks, dtype=float)
        self.tau = 0.01
        self.fitted = True
        self.risk_model = None
        self.platt = None
    def score(self, surrogate, u_t, params, t, dt):
        return float(self.risks[0])  # placeholder, not used by ranked directly


def test_budget_ranked_respects_budget():
    from controller.policies import Policy, ControlContext, Controller
    nx = 32
    n_steps = 40
    # distinct risks so ranking is meaningful
    risks = np.linspace(0.0, 1.0, n_steps)
    re = _MockRiskEstimator(risks)
    solver = _MockSolver()
    surrogate = _MockSurrogate()
    u0 = np.zeros(nx)
    params = np.array([0.02])
    t0 = 0.0; dt = 0.001
    for budget in [0.05, 0.2, 0.6]:
        ctx = ControlContext(surrogate=surrogate, solver=solver, risk_estimator=re,
                             L=1.0, dt=dt, budget=budget, threshold=0.5,
                             target_intervention_rate=budget)
        c = Controller(Policy.BUDGET_RISK_RANKED, ctx)
        # monkeypatch score to return the precomputed risks in order
        call = {"i": 0}
        def fake_score(surr, u_t, p, tt, ddt):
            r = float(risks[call["i"]])
            call["i"] += 1
            return r
        re.score = fake_score
        c.prepare(u0, params, t0, dt, n_steps)
        n_allowed = int(round(budget * n_steps))
        n_sel = len(c.ctx.intervention_indices)
        assert n_sel <= n_allowed + 1, (budget, n_sel, n_allowed)
