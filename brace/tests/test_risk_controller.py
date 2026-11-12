"""Tests for risk calibration, budget allocation, correction and statistics."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.calibration.conformal import ConformalError, conformal_quantile
from src.controller.policies import AdaptiveBudgetController, topk_actions
from src.evaluation.stats import paired_ci, summarize_seeds
from src.controller.rollout import correction_diagnostics
from src.uncertainty.risk import RiskModel, auroc
from src.uncertainty.features import step_error


def test_auroc_perfect_and_random():
    y = np.array([0, 0, 1, 1])
    assert auroc(np.array([0.1, 0.2, 0.8, 0.9]), y) == 1.0
    assert auroc(np.array([0.9, 0.8, 0.2, 0.1]), y) == 0.0


def test_risk_model_learns_separable_signal():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 9))
    err = 0.001 + 0.2 / (1 + np.exp(-3 * X[:, 2])) + 0.002 * rng.random(400)
    rm = RiskModel(tau_fail=0.05).fit(X[:300], err[:300])
    rep = rm.report(X[300:], err[300:])
    assert rep["auroc"] > 0.9
    assert rep["brier"] < 0.15


def test_conformal_coverage_synthetic():
    rng = np.random.default_rng(1)
    n = 2000
    log_pred = rng.normal(size=n)
    errors = np.exp(log_pred + rng.normal(scale=0.2, size=n))
    c = ConformalError(alpha=0.1).fit(log_pred[:1000], errors[:1000])
    cov = c.coverage(log_pred[1000:], errors[1000:])
    assert 0.85 < cov < 0.95
    assert c.upper_coverage(log_pred[1000:], errors[1000:]) > 0.93


def test_conformal_quantile_index():
    assert conformal_quantile(np.arange(1, 10), 0.1) == 9.0
    assert np.isinf(conformal_quantile(np.array([]), 0.1))


def test_topk_budget_exact_and_spacing():
    scores = np.random.default_rng(0).random((5, 100))
    a = topk_actions(scores, 0.1)
    assert a.shape == (5, 100)
    assert np.all(a.sum(axis=1) == 10)
    # selected are the highest scores
    for i in range(5):
        selected = np.where(a[i])[0]
        unselected = np.where(~a[i])[0]
        assert scores[i, selected].min() >= scores[i, unselected].max()

    b = topk_actions(scores, 0.2, cost_per_action=5, min_spacing=5)
    for i in range(5):
        assert np.all(b[i].sum() == 4)
        idx = np.where(b[i])[0]
        assert np.all(np.diff(idx) >= 5)

    # Window cost may not be truncated at the episode start.
    c = topk_actions(scores, 0.2, cost_per_action=5, min_spacing=5, min_start=4)
    for i in range(5):
        idx = np.where(c[i])[0]
        assert np.all(idx >= 4)
        assert np.all(c[i].sum() == 4)


def test_adaptive_controller_respects_budget():
    ctrl = AdaptiveBudgetController(n_traj=2, horizon=100, budget=0.1)
    rng = np.random.default_rng(0)
    spent = np.zeros(2)
    for t in range(100):
        scores = rng.random(2)
        act = ctrl.decide(t, scores)
        spent += act
    assert np.all(spent <= 100 * 0.1 + 100 * 0.05 * 0.1 + 1)


def test_step_error_zero_and_scale():
    a = np.ones((3, 16))
    assert np.allclose(step_error(a, a), 0.0)
    b = a.copy()
    b[:, 0] += 1.0
    e = step_error(a, b)
    assert np.all(e > 0)


def test_paired_ci_identical():
    a = np.arange(20, dtype=float)
    r = paired_ci(a, a.copy())
    assert r["mean_diff"] == 0.0
    assert r["p_value"] == 1.0


def test_summarize_seeds_uses_t_interval():
    s = summarize_seeds([1.0, 2.0])
    # Student t with 1 dof: 12.706 * 0.7071 / sqrt(2)
    assert s["ci95"] == pytest.approx(6.353, rel=1e-3)
    assert s["ci95"] > 1.96 * 0.7071 / np.sqrt(2)


def test_correction_diagnostics_sign_and_profiles():
    rng = np.random.default_rng(0)
    errs = rng.random((4, 20)) + 0.01
    pre = errs + 0.05
    actions = np.zeros((4, 20), dtype=bool)
    actions[:, :10] = True
    d = correction_diagnostics(errs, pre, actions)
    assert d["correction_gain_mean"] == pytest.approx(0.05, abs=1e-6)
    assert d["correction_harm_rate"] == 0.0
    assert len(d["correction_gain_profile"]) == 20
    assert d["correction_by_time_profile"][0] == 1.0
    assert d["correction_by_time_profile"][-1] == 0.0

    none = correction_diagnostics(errs, pre, np.zeros((4, 20), dtype=bool))
    assert np.isnan(none["correction_gain_mean"])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
