"""Measure how the value of a numerical correction depends on drift.

The budgeted-correction results show the surprising pattern that a trivial
"correct the first k steps" schedule matches or beats risk-ranked allocation at
large budgets. A plausible mechanism is that a correction integrates the PDE
*from the surrogate state*: once that state has drifted away from the reference
solution, one saved interval of the true solver no longer lands near the truth,
so the correction stops helping and can even hurt.

This script measures that directly. For an uncorrected rollout it evaluates, at
every step, the counterfactual one-step error with and without a correction, and
reports

  * relative gain = (e_neural - e_corrected) / e_neural  by horizon position,
  * the harm rate (fraction of steps where the correction increases the error),
  * the Spearman correlation between the pre-correction error and the gain.

Everything is computed on held-out trajectories; the script is a diagnostic and
does not train anything.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from src.controller.rollout import counterfactual_gains, neural_pass
from src.models.fno import build_model
from src.uncertainty.features import step_error
from src.uncertainty.pipeline import filter_params, residual_solver


def load_split(benchmark: str, split: str, downsample_to: int = 0) -> np.ndarray:
    z = np.load(f"data/processed/{benchmark}_{split}.npz")
    u = z["u"]
    if downsample_to:
        uh = np.fft.rfft2(u, axes=(-2, -1))[..., :downsample_to, : downsample_to // 2 + 1]
        u = np.fft.irfft2(uh, s=(downsample_to, downsample_to), axes=(-2, -1))
    return u.astype(np.float32)


def load_params(benchmark: str, split: str) -> dict:
    z = np.load(f"data/processed/{benchmark}_{split}.npz")
    return {k: z[k] for k in ("nu", "cx", "cy", "eps") if k in z}


def build_models(meta: dict) -> list:
    cfg = meta["config"]
    models = []
    for rec in meta["members"]:
        if meta["model"] == "fno2d":
            m = build_model("fno2d", width=cfg["width"], modes1=cfg["modes"],
                            modes2=cfg["modes"], layers=cfg["layers"])
        elif meta["model"] == "deeponet1d":
            m = build_model("deeponet1d", n_points=128)
        else:
            m = build_model("fno1d", width=cfg["width"], modes=cfg["modes"],
                            layers=cfg["layers"])
        ck = torch.load(f"checkpoints/{meta['tag']}_m{rec['member']}.pt",
                        map_location="cpu", weights_only=False)
        m.load_state_dict(ck["state_dict"])
        m.eval()
        models.append(m)
    return models


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    from scipy import stats
    v = np.isfinite(x) & np.isfinite(y)
    if v.sum() < 3:
        return float("nan")
    return float(stats.spearmanr(x[v], y[v]).statistic)


def analyse(benchmark: str, tag: str, split: str, n_traj: int, n_bins: int,
            downsample_to: int, threads: int) -> dict:
    torch.set_num_threads(threads)
    meta = json.load(open(f"results/raw/train_{tag}.json"))
    meta["tag"] = tag
    models = build_models(meta)
    u_full = load_split(benchmark, split, downsample_to)
    u = u_full[:n_traj]
    params = {k: v[:n_traj] for k, v in load_params(benchmark, split).items()}
    dt = float(np.load(f"data/processed/{benchmark}_{split}.npz")["dt_save"])

    solver = residual_solver(benchmark, dt, n=u.shape[-1])
    feats, errs, states = neural_pass(models, u, solver, filter_params(benchmark, params),
                                      dt, store_states=True)
    gains = counterfactual_gains(states, u, benchmark, params, dt)
    pre = np.zeros_like(gains)
    for t in range(u.shape[1] - 1):
        pre[:, t] = step_error(states[:, t + 1], u[:, t + 1])

    T = gains.shape[1]
    rel = gains / (pre + 1e-12)
    bins = np.linspace(0, 1, n_bins + 1)
    pos = (np.arange(T) + 0.5) / T
    rows = []
    for j in range(n_bins):
        m = (pos >= bins[j]) & (pos < bins[j + 1])
        if not m.any():
            continue
        rows.append({
            "bin": j, "t_start": int(bins[j] * T), "t_end": int(bins[j + 1] * T),
            "mean_pre_error": float(pre[:, m].mean()),
            "mean_gain": float(gains[:, m].mean()),
            "mean_relative_gain": float(rel[:, m].mean()),
            "median_relative_gain": float(np.median(rel[:, m])),
            "harm_rate": float((gains[:, m] < 0).mean()),
            "zero_gain_rate": float((np.abs(rel[:, m]) < 1e-3).mean()),
        })

    # Does the pre-error predict whether a correction helps?
    rho = spearman(pre.reshape(-1), rel.reshape(-1))
    rho_abs = spearman(pre.reshape(-1), gains.reshape(-1))
    early = rel[:, : max(1, T // 4)]
    late = rel[:, 3 * T // 4:]
    return {
        "benchmark": benchmark, "tag": tag, "split": split, "n_traj": int(len(u)),
        "horizon": int(T),
        "mean_pre_error": float(pre.mean()),
        "mean_gain": float(gains.mean()),
        "mean_relative_gain": float(rel.mean()),
        "harm_rate": float((gains < 0).mean()),
        "relative_gain_early": float(early.mean()),
        "relative_gain_late": float(late.mean()),
        "spearman_pre_vs_relative_gain": rho,
        "spearman_pre_vs_absolute_gain": rho_abs,
        "bins": rows,
    }


def write_outputs(out: list[dict], out_path: str) -> None:
    """Persist the diagnostic as JSON, a flat CSV and one figure per benchmark."""
    import csv

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"runs": out}, f, indent=2)

    rows = []
    for r in out:
        for b in r["bins"]:
            rows.append({"benchmark": r["benchmark"], "split": r["split"],
                         "tag": r["tag"], "bin": b["bin"],
                         "mean_pre_error": b["mean_pre_error"],
                         "mean_gain": b["mean_gain"],
                         "mean_relative_gain": b["mean_relative_gain"],
                         "harm_rate": b["harm_rate"]})
    if rows:
        os.makedirs("results/tables", exist_ok=True)
        path = "results/tables/gain_decay.csv"
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    os.makedirs("results/figures", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for r in out:
        bins = [b["bin"] for b in r["bins"]]
        axes[0].plot(bins, [b["mean_relative_gain"] for b in r["bins"]],
                     marker="o", ms=3, label=f"{r['benchmark']}/{r['split']}")
        axes[1].plot(bins, [b["harm_rate"] for b in r["bins"]],
                     marker="o", ms=3, label=f"{r['benchmark']}/{r['split']}")
    axes[0].set_xlabel("horizon decile")
    axes[0].set_ylabel("mean relative gain")
    axes[0].set_title("Value of a correction decays with horizon position")
    axes[1].set_xlabel("horizon decile")
    axes[1].set_ylabel("fraction of harmful corrections")
    axes[1].set_title("Harm rate rises late in the rollout")
    for ax in axes:
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig("results/figures/gain_decay.png", dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/processed/gain_decay.json")
    ap.add_argument("--n-traj", type=int, default=40)
    ap.add_argument("--n-bins", type=int, default=10)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--jobs", default="all",
                    help="'all' or a comma-separated list of tag names")
    args = ap.parse_args()

    jobs = [
        ("burgers1d", "burgers1d_fno1d_s0", "test", 0),
        ("burgers1d", "burgers1d_fno1d_s0", "ood_param", 0),
        ("allencahn1d", "allencahn1d_fno1d_s0", "test", 0),
        ("advection2d", "advection2d_fno2d_s0", "test", 32),
    ]
    if args.jobs != "all":
        keep = set(args.jobs.split(","))
        jobs = [j for j in jobs if j[1] in keep or f"{j[1]}:{j[2]}" in keep]

    out = []
    for bench, tag, split, ds in jobs:
        if not os.path.exists(f"results/raw/train_{tag}.json"):
            print(f"[skip] {tag}: no manifest")
            continue
        if not os.path.exists(f"results/raw/main/{bench}_{split}_s0_step.json"):
            print(f"[skip] {bench}/{split}: no main results")
            continue
        print(f"== {tag} / {split} ==", flush=True)
        r = analyse(bench, tag, split, args.n_traj, args.n_bins, ds, args.threads)
        out.append(r)
        print(f"   relative gain early {r['relative_gain_early']:+.3f} -> late "
              f"{r['relative_gain_late']:+.3f}; harm rate {r['harm_rate']:.3f}; "
              f"rho(pre, rel gain)={r['spearman_pre_vs_relative_gain']:+.3f}", flush=True)

    write_outputs(out, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
