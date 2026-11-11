"""Statistical utilities for paired experiment comparisons."""
from __future__ import annotations

import numpy as np
from scipy import stats


def paired_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 5000, seed: int = 0,
              alpha: float = 0.05) -> dict:
    """Paired comparison a vs b over identical trajectories.

    Returns mean difference (a-b), bootstrap CI, paired t-test, Wilcoxon,
    and Cohen's d_z.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    assert a.shape == b.shape
    d = a - b
    n = len(d)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = d[idx].mean(axis=1)
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    t = stats.ttest_rel(a, b)
    try:
        w = stats.wilcoxon(a, b)
        wp = float(w.pvalue)
    except ValueError:
        wp = float("nan")
    dz = float(d.mean() / (d.std(ddof=1) + 1e-12))
    if np.allclose(d, 0.0):
        return {
            "mean_a": float(a.mean()), "mean_b": float(b.mean()),
            "mean_diff": 0.0, "ci_low": 0.0, "ci_high": 0.0,
            "t_stat": 0.0, "p_value": 1.0, "wilcoxon_p": 1.0,
            "cohen_dz": 0.0, "n": int(n), "ci95": 0.0,
        }
    return {
        "mean_a": float(a.mean()), "mean_b": float(b.mean()),
        "mean_diff": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi),
        "t_stat": float(t.statistic), "p_value": float(t.pvalue),
        "wilcoxon_p": wp, "cohen_dz": dz, "n": int(n),
        "ci95": float(1.96 * d.std(ddof=1) / np.sqrt(n)),
    }


def summarize_seeds(values: list[float]) -> dict:
    v = np.asarray(values, dtype=np.float64)
    n = len(v)
    sd = float(v.std(ddof=1)) if n > 1 else 0.0
    half = float(1.96 * sd / np.sqrt(n)) if n > 1 else 0.0
    return {"mean": float(v.mean()), "std": sd, "ci95": half, "n_seeds": n,
            "values": v.tolist()}
