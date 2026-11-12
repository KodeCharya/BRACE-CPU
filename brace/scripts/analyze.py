"""Aggregate experiment records into tables, statistics and figures."""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
from scipy import stats

from src.evaluation.stats import paired_ci, summarize_seeds

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_records(exp_id: str):
    recs = []
    for path in sorted(glob.glob(f"results/raw/{exp_id}/*.json")):
        if path.endswith("_risk.json"):
            continue
        with open(path) as f:
            blob = json.load(f)
        for r in blob["records"]:
            r["benchmark"] = blob["config"]["benchmark"]
            r["split"] = blob["config"]["split"]
            r["seed"] = blob["config"]["seed"]
            r["exp_id"] = exp_id
            recs.append(r)
    return recs


def load_risk(exp_id: str):
    out = []
    for path in sorted(glob.glob(f"results/raw/{exp_id}/*_risk.json")):
        with open(path) as f:
            b = json.load(f)
        b["exp_id"] = exp_id
        out.append(b)
    return out


def write_csv(path: str, rows: list[dict], fields: list[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(",".join(fields) + "\n")
        for r in rows:
            vals = []
            for k in fields:
                v = r.get(k, "")
                if isinstance(v, float):
                    vals.append(f"{v:.6g}")
                else:
                    vals.append(str(v))
            f.write(",".join(vals) + "\n")


def _nanmean(vals):
    vals = [float(v) for v in vals if v == v]
    return float(np.mean(vals)) if vals else float("nan")


def aggregate(recs):
    groups = defaultdict(list)
    for r in recs:
        groups[(r["benchmark"], r["split"], r.get("variant", ""), r["policy"], r["budget"])].append(r)
    rows = []
    for (bench, split, variant, policy, budget), rs in sorted(groups.items(),
                                                     key=lambda kv: (kv[0][0], kv[0][1], kv[0][2], kv[0][3], -1 if kv[0][4] is None else kv[0][4])):
        rows.append({
            "benchmark": bench, "split": split, "variant": variant, "policy": policy,
            "budget": "" if budget is None else budget,
            "n_seeds": len(rs),
            "rel_l2_mean": float(np.mean([r["rel_l2_mean"] for r in rs])),
            "rel_l2_seed_min": float(np.min([r["rel_l2_mean"] for r in rs])),
            "rel_l2_seed_max": float(np.max([r["rel_l2_mean"] for r in rs])),
            "rel_l2_median": float(np.mean([r.get("rel_l2_median", r["rel_l2_mean"]) for r in rs])),
            "divergence_rate": float(np.mean([r.get("divergence_rate", 0.0) for r in rs])),
            # Seed-level t interval; with two seeds the multiplier is large and
            # the interval is reported as an honest lower bound on uncertainty.
            "rel_l2_ci95": (float(stats.t.ppf(0.975, len(rs) - 1)
                                  * np.std([r["rel_l2_mean"] for r in rs], ddof=1)
                                  / np.sqrt(len(rs))) if len(rs) > 1 else 0.0),
            "rel_l2_final": float(np.mean([r["rel_l2_final"] for r in rs])),
            "rel_l2_max": float(np.mean([r["rel_l2_max"] for r in rs])),
            "step_failure_rate": float(np.mean([r["step_failure_rate"] for r in rs])),
            "traj_failure_rate": float(np.mean([r["traj_failure_rate"] for r in rs])),
            "cost_fraction": float(np.mean([r["cost_fraction"] for r in rs])),
            "intervention_fraction": float(np.mean([r["intervention_fraction"] for r in rs])),
            "wall_seconds": float(np.mean([r.get("wall_seconds", 0) for r in rs])),
            "plan_seconds": float(np.mean([r.get("plan_seconds", 0) for r in rs])),
            "total_seconds": float(np.mean([r.get("total_seconds", r.get("wall_seconds", 0))
                                            for r in rs])),
            "pre_rel_l2_mean": float(np.mean([r.get("pre_rel_l2_mean", float("nan")) for r in rs])),
            "action_time_mean": float(np.mean([r.get("action_time_mean", float("nan")) for r in rs])),
            "correction_gain_mean": _nanmean([r.get("correction_gain_mean", float("nan")) for r in rs]),
            "correction_harm_rate": _nanmean([r.get("correction_harm_rate", float("nan")) for r in rs]),
        })
    return rows


def paired_analysis(recs):
    # index by (bench, split, seed, budget) -> policy -> per-traj means
    idx = defaultdict(dict)
    for r in recs:
        key = (r["benchmark"], r["split"], r.get("variant", ""), r["seed"], r["budget"])
        idx[key][r["policy"]] = np.asarray(r["per_traj_mean"])
    rows = []
    for bench in sorted({k[0] for k in idx}):
        for split in sorted({k[1] for k in idx if k[0] == bench}):
          for variant in sorted({k[2] for k in idx if k[0] == bench and k[1] == split}):
            for budget in sorted({k[4] for k in idx if k[0] == bench and k[1] == split and k[2] == variant},
                                 key=lambda b: -1 if b is None else b):
                pairs = [
                    ("brace", "neural"), ("brace", "fixed"), ("brace", "ranked_residual"),
                    ("brace", "ranked_calibrated"), ("brace", "adaptive"),
                    ("brace", "oracle"), ("brace", "uniform"), ("brace", "early"),
                    ("brace", "ranked_ensemble"),
                    ("brace_combo", "neural"), ("brace_combo", "brace"),
                    ("brace_combo", "ranked_ensemble"), ("brace_combo", "early"),
                    ("ranked_ensemble", "neural"), ("uniform", "neural"),
                    ("early", "neural"), ("ranked_residual", "fixed"),
                    ("neural", "fixed"),
                    ("gain_ranked", "neural"), ("gain_ranked", "ranked_ensemble"),
                    ("gain_ranked", "early"), ("gain_ranked", "brace"),
                    ("oracle_gain", "early"), ("oracle_gain", "ranked_ensemble"),
                ]
                for a, b in pairs:
                    diffs, ps, ds = [], [], []
                    for seed in sorted({k[3] for k in idx if k[0] == bench and k[1] == split and k[2] == variant and k[4] == budget}):
                        d = idx[(bench, split, variant, seed, budget)]
                        if a in d and b in d and len(d[a]) == len(d[b]):
                            st = paired_ci(d[a], d[b])
                            diffs.append(st["mean_diff"])
                            ps.append(st["p_value"])
                            ds.append(st["cohen_dz"])
                    if diffs:
                        s = summarize_seeds(diffs)
                        rows.append({
                            "benchmark": bench, "split": split, "variant": variant,
                            "budget": "" if budget is None else budget,
                            "comparison": f"{a} vs {b}",
                            "mean_diff": s["mean"],
                            "ci95_across_seeds": s["ci95"],
                            "n_seeds": s["n_seeds"],
                            "cohen_dz_mean": float(np.mean(ds)),
                            "p_min": float(np.min(ps)),
                            "p_max": float(np.max(ps)),
                            "seeds_sig_005": int(np.sum(np.asarray(ps) < 0.05)),
                        })
    return rows


def plot_results(agg, exp_id="main", out_dir="results/figures"):
    os.makedirs(out_dir, exist_ok=True)
    for bench in sorted({r["benchmark"] for r in agg}):
        for split in sorted({r["split"] for r in agg if r["benchmark"] == bench}):
            sub = [r for r in agg if r["benchmark"] == bench and r["split"] == split
                   and r["budget"] != ""]
            if not sub:
                continue
            fig, ax = plt.subplots(figsize=(6, 4))
            for policy in sorted({r["policy"] for r in sub}):
                rows = sorted([r for r in sub if r["policy"] == policy], key=lambda r: r["budget"])
                if not rows:
                    continue
                x = [r["cost_fraction"] for r in rows]
                y = [r["rel_l2_mean"] for r in rows]
                ax.plot(x, y, marker="o", label=policy)
            base = [r for r in agg if r["benchmark"] == bench and r["split"] == split
                    and r["policy"] == "neural"]
            if base:
                ax.axhline(base[0]["rel_l2_mean"], ls="--", c="k", lw=1, label="neural-only")
            ax.set_xlabel("solver cost fraction")
            ax.set_ylabel("mean relative L2")
            ax.set_title(f"{bench} / {split}")
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(f"{out_dir}/{exp_id}_pareto_{bench}_{split}.png", dpi=150)
            plt.close(fig)

    # Risk-coverage curves per benchmark/split
    risk = load_risk(exp_id)
    by = defaultdict(list)
    for b in risk:
        by[(b["benchmark"], b["split"])].append(b)
    for (bench, split), rs in sorted(by.items()):
        fig, ax = plt.subplots(figsize=(5, 4))
        for key, label in [("risk_coverage", "BRACE hybrid"),
                           ("risk_coverage_residual", "residual"),
                           ("risk_coverage_ensemble", "ensemble std")]:
            curves = np.array([r["risk_eval"][key]["mean_error"] for r in rs if key in r["risk_eval"]])
            if len(curves) == 0:
                continue
            cov = np.array(rs[0]["risk_eval"][key]["coverage"])
            m = curves.mean(axis=0)
            sd = curves.std(axis=0, ddof=1) if len(curves) > 1 else np.zeros_like(m)
            ax.plot(cov, m, marker=".", label=label)
            ax.fill_between(cov, m - sd, m + sd, alpha=0.15)
        neural = [r["rel_l2_mean"] for r in agg if r["benchmark"] == bench and r["split"] == split
                  and r["policy"] == "neural"]
        if neural:
            ax.axhline(neural[0], ls="--", c="k", lw=1, label="neural-only")
        ax.set_xlabel("coverage (fraction of steps accepted)")
        ax.set_ylabel("mean relative L2 of accepted steps")
        ax.set_title(f"risk-coverage: {bench}/{split}")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"{out_dir}/{exp_id}_risk_coverage_{bench}_{split}.png", dpi=150)
        plt.close(fig)

    # Reliability diagrams
    by = defaultdict(list)
    for b in risk:
        by[(b["benchmark"], b["split"])].append(b)
    for (bench, split), rs in sorted(by.items()):
        fig, ax = plt.subplots(figsize=(4.2, 4))
        centers = np.linspace(0.05, 0.95, 10)
        mp, ob = [], []
        for c in centers:
            vals = []
            for r in rs:
                for row in r["risk_eval"]["reliability"]["bins"]:
                    if abs(row["bin_center"] - c) < 1e-6:
                        vals.append((row["mean_pred"], row["observed"]))
            if vals:
                mp.append(np.mean([v[0] for v in vals]))
                ob.append(np.mean([v[1] for v in vals]))
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.plot(mp, ob, "o-", label="risk model")
        ax.set_xlabel("predicted failure probability")
        ax.set_ylabel("observed failure frequency")
        ax.set_title(f"reliability: {bench}/{split}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"{out_dir}/{exp_id}_reliability_{bench}_{split}.png", dpi=150)
        plt.close(fig)

    # Long-horizon error curves
    exp = load_records(exp_id)
    for bench in sorted({r["benchmark"] for r in exp}):
        for split in sorted({r["split"] for r in exp if r["benchmark"] == bench}):
            sub = [r for r in exp if r["benchmark"] == bench and r["split"] == split]
            fig, ax = plt.subplots(figsize=(6, 4))
            plotted = 0
            for policy in ["neural", "fixed", "ranked_residual", "brace", "numerical"]:
                rows = [r for r in sub if r["policy"] == policy]
                if not rows:
                    continue
                curves = np.array([r["per_step_mean"] for r in rows])
                ax.plot(curves.mean(axis=0), label=f"{policy} (n={len(rows)})")
                plotted += 1
            if plotted:
                ax.set_xlabel("saved timestep")
                ax.set_ylabel("relative L2")
                ax.set_yscale("log")
                ax.set_title(f"error over horizon: {bench}/{split}")
                ax.legend(fontsize=7)
                ax.grid(alpha=0.3)
                fig.tight_layout()
                fig.savefig(f"{out_dir}/{exp_id}_horizon_{bench}_{split}.png", dpi=150)
            plt.close(fig)


def plot_budget_fidelity(agg, exp_id, out_dir="results/figures"):
    """Declared budget versus realised solver cost."""
    os.makedirs(out_dir, exist_ok=True)
    for bench in sorted({r["benchmark"] for r in agg}):
        sub = [r for r in agg if r["benchmark"] == bench and r["budget"] != ""
               and r.get("variant", "") == ""]
        if not sub:
            continue
        policies = sorted({r["policy"] for r in sub
                           if r["policy"] in ("fixed", "adaptive", "uniform", "early",
                                              "ranked_residual", "ranked_ensemble",
                                              "ranked_calibrated", "brace", "brace_combo",
                                              "oracle", "reanchor")})
        fig, ax = plt.subplots(figsize=(5, 4.2))
        lim = max(float(r["budget"]) for r in sub) * 1.15
        ax.plot([0, lim], [0, lim], "k--", lw=1, label="declared = realised")
        for pol in policies:
            rows = sorted([r for r in sub if r["policy"] == pol], key=lambda r: float(r["budget"]))
            if not rows:
                continue
            ax.plot([float(r["budget"]) for r in rows], [r["cost_fraction"] for r in rows],
                    marker="o", label=pol)
        ax.set_xlabel("declared budget $B$")
        ax.set_ylabel("realised solver cost fraction")
        ax.set_title(f"budget fidelity: {bench}")
        ax.legend(fontsize=6, ncol=2)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"{out_dir}/{exp_id}_budget_fidelity_{bench}.png", dpi=150)
        plt.close(fig)


def plot_bar_at_budget(agg, exp_id, budget=0.2, out_dir="results/figures"):
    for bench in sorted({r["benchmark"] for r in agg}):
        rows = [r for r in agg if r["benchmark"] == bench
                and r.get("variant", "") == ""
                and r["budget"] not in ("",)
                and abs(float(r["budget"]) - budget) < 1e-9]
        if not rows:
            continue
        rows = sorted(rows, key=lambda r: r["rel_l2_mean"])
        fig, ax = plt.subplots(figsize=(7, 3.6))
        x = np.arange(len(rows))
        vals = np.array([r["rel_l2_mean"] for r in rows])
        lo = np.array([r["rel_l2_mean"] - r.get("rel_l2_seed_min", r["rel_l2_mean"]) for r in rows])
        hi = np.array([r.get("rel_l2_seed_max", r["rel_l2_mean"]) - r["rel_l2_mean"] for r in rows])
        ax.bar(x, vals, yerr=np.vstack([lo, hi]), capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels([r["policy"] for r in rows], rotation=35, ha="right", fontsize=7)
        ax.set_ylabel("mean relative $L_2$")
        ax.set_title(f"{bench}: declared budget {budget:g}")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"{out_dir}/{exp_id}_bars_B{int(budget*100)}_{bench}.png", dpi=150)
        plt.close(fig)


def export_correction_gain(exp_id: str = "main", budget: float = 0.2,
                           out_dir: str = "results/figures"):
    """Correction gain (pre-error minus post-error) as a function of horizon position.

    Averaged over test trajectories and seeds. A positive value means the
    numerical correction reduced the one-step error at that step.
    """
    recs = load_records(exp_id)
    rows = []
    for bench in sorted({r["benchmark"] for r in recs}):
        sub = [r for r in recs if r["benchmark"] == bench
               and "correction_gain_profile" in r
               and r.get("budget") is not None
               and abs(float(r["budget"]) - budget) < 1e-9
               and r.get("variant", "") == ""]
        if not sub:
            continue
        fig, ax = plt.subplots(figsize=(6, 3.8))
        for pol in ["early", "uniform", "ranked_ensemble", "brace", "brace_combo"]:
            rs = [r for r in sub if r["policy"] == pol]
            if not rs:
                continue
            prof = np.array([r["correction_gain_profile"] for r in rs], dtype=float)
            act = np.array([r["correction_by_time_profile"] for r in rs], dtype=float)
            ax.plot(np.nanmean(prof, axis=0), marker=".", ms=3, label=pol)
            for j in range(prof.shape[1]):
                rows.append({"benchmark": bench, "budget": budget, "policy": pol,
                             "bin": j, "mean_gain": float(np.nanmean(prof[:, j]))
                             if np.any(~np.isnan(prof[:, j])) else "",
                             "action_rate": float(act[:, j].mean())})
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlabel("normalised horizon position (20 bins)")
        ax.set_ylabel("mean error reduction per correction")
        ax.set_title(f"{bench}: local correction gain at B={budget:g}")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"{out_dir}/{exp_id}_correction_gain_{bench}_B{int(budget*100)}.png", dpi=150)
        plt.close(fig)
    if rows:
        write_csv(f"results/tables/{exp_id}_correction_gain.csv", rows,
                  ["benchmark", "budget", "policy", "bin", "mean_gain", "action_rate"])


def summary_at_budget(agg, exp_id: str):
    """Compact per-benchmark comparison at each declared budget."""
    rows = []
    for r in agg:
        if r["budget"] == "":
            continue
        rows.append({"variant": r.get("variant", ""),
                     **{k: r[k] for k in ("benchmark", "split", "policy", "budget", "n_seeds",
                                          "rel_l2_mean", "rel_l2_median", "divergence_rate",
                                          "rel_l2_seed_min", "rel_l2_seed_max",
                                          "rel_l2_ci95", "cost_fraction", "step_failure_rate",
                                          "correction_gain_mean", "correction_harm_rate")}})
    write_csv(f"results/tables/{exp_id}_at_budget.csv", rows,
              ["benchmark", "split", "variant", "policy", "budget", "n_seeds", "rel_l2_mean",
               "rel_l2_median", "divergence_rate",
               "rel_l2_seed_min", "rel_l2_seed_max", "rel_l2_ci95", "cost_fraction",
               "step_failure_rate", "correction_gain_mean", "correction_harm_rate"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", default="main")
    args = ap.parse_args()
    recs = load_records(args.exp_id)
    if not recs:
        raise SystemExit(f"no records for exp {args.exp_id}")
    agg = aggregate(recs)
    write_csv(f"results/tables/{args.exp_id}_aggregate.csv", agg,
              ["benchmark", "split", "variant", "policy", "budget", "n_seeds", "rel_l2_mean",
               "rel_l2_seed_min", "rel_l2_seed_max", "rel_l2_median", "divergence_rate",
               "rel_l2_ci95", "rel_l2_final",
               "rel_l2_max", "step_failure_rate",
               "traj_failure_rate", "cost_fraction", "intervention_fraction",
               "pre_rel_l2_mean", "action_time_mean", "correction_gain_mean",
               "correction_harm_rate", "wall_seconds",
               "plan_seconds", "total_seconds"])
    pairs = paired_analysis(recs)
    write_csv(f"results/tables/{args.exp_id}_paired.csv", pairs,
              ["benchmark", "split", "variant", "budget", "comparison", "mean_diff",
               "ci95_across_seeds", "n_seeds", "cohen_dz_mean", "p_min", "p_max",
               "seeds_sig_005"])

    risk = load_risk(args.exp_id)
    rrows = []
    for b in risk:
        e = b["risk_eval"]
        rrows.append({
            "benchmark": b["benchmark"], "split": b["split"], "seed": b["seed"],
            "n_test": b["n_test"], "failure_rate": e["failure_rate"],
            "auroc_residual": e["auroc_residual"], "auroc_ensemble": e["auroc_ensemble"],
            "auroc_calibrated": e["auroc_calibrated"], "auroc_brace": e["auroc_brace"],
            "ece": e["ece"], "brier": e["brier"],
            "conformal_coverage": e["conformal_coverage"],
            "conformal_upper_coverage": e["conformal_upper_coverage"],
            "mean_pred_log_error_bias": "",
        })
    write_csv(f"results/tables/{args.exp_id}_risk.csv", rrows,
              ["benchmark", "split", "seed", "n_test", "failure_rate", "auroc_residual",
               "auroc_ensemble", "auroc_calibrated", "auroc_brace", "ece", "brier",
               "conformal_coverage", "conformal_upper_coverage"])

    # Feature AUROC aggregation
    feat_rows = []
    features = None
    for b in risk:
        af = b["risk_eval"].get("auroc_features", {})
        if af:
            features = list(af.keys())
            break
    if features:
        for f in features:
            vals = [b["risk_eval"]["auroc_features"][f] for b in risk
                    if f in b["risk_eval"].get("auroc_features", {})]
            feat_rows.append({"feature": f, "mean_auroc": float(np.mean(vals)),
                              "std_auroc": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                              "n": len(vals)})
        write_csv(f"results/tables/{args.exp_id}_feature_auroc.csv", feat_rows,
                  ["feature", "mean_auroc", "std_auroc", "n"])

    plot_results(agg, args.exp_id)
    plot_budget_fidelity(agg, args.exp_id)
    for b in (0.1, 0.2, 0.4):
        plot_bar_at_budget(agg, args.exp_id, b)
    export_correction_gain(args.exp_id, budget=0.2)
    summary_at_budget(agg, args.exp_id)
    with open(f"results/processed/{args.exp_id}_summary.json", "w") as f:
        json.dump({"aggregate": agg, "paired": pairs, "risk": rrows}, f, indent=2)
    print(f"aggregated {len(recs)} records -> {len(agg)} groups, {len(pairs)} comparisons")


if __name__ == "__main__":
    main()
