"""Analyze BRACE-CPU experiment results.

Produces summary tables (CSV) and figures (Pareto accuracy-vs-intervention,
accuracy-vs-budget) from raw result records.
Run from project root:
    PYTHONPATH=src .venv/bin/python scripts/analyze_results.py <results_raw_dir>
"""

from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as e:
    print("matplotlib unavailable:", e)
    plt = None


def ci95(x):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return 0.0
    return float(np.std(x, ddof=1) / np.sqrt(len(x)) * 1.96)


def load_records(raw_dir):
    recs = []
    for path in glob.glob(os.path.join(raw_dir, "*_all.json")):
        with open(path) as f:
            recs.extend(json.load(f))
    # also pick up any loose per-record jsons not in an _all.json
    alljson = {os.path.basename(p) for p in glob.glob(os.path.join(raw_dir, "*_all.json"))}
    for path in glob.glob(os.path.join(raw_dir, "*.json")):
        if os.path.basename(path) in alljson:
            continue
        try:
            with open(path) as f:
                r = json.load(f)
            if "policy" in r:
                recs.append(r)
        except Exception:
            pass
    return recs


def summarize(recs):
    groups = defaultdict(list)
    for r in recs:
        groups[r["policy"]].append(r)
    rows = []
    for pol in sorted(groups):
        rs = groups[pol]
        rel = np.array([r["rel_l2_mean"] for r in rs])
        frac = np.array([r["intervention_fraction"] for r in rs])
        sw = np.array([r["solver_wall_s"] for r in rs])
        rows.append({
            "policy": pol,
            "n": len(rs),
            "rel_l2_mean": float(rel.mean()),
            "rel_l2_std": float(rel.std(ddof=1)) if len(rel) > 1 else 0.0,
            "rel_l2_ci95": ci95(rel),
            "rel_l2_max_mean": float(np.array([r["rel_l2_max"] for r in rs]).mean()),
            "intervention_fraction_mean": float(frac.mean()),
            "intervention_fraction_std": float(frac.std(ddof=1)) if len(frac) > 1 else 0.0,
            "solver_wall_s_mean": float(sw.mean()),
            "total_wall_s_mean": float(np.array([r["total_wall_s"] for r in rs]).mean()),
        })
    return rows


def summarize_by_budget(recs, policy):
    groups = defaultdict(list)
    for r in recs:
        if r["policy"] == policy:
            groups[r["budget"]].append(r)
    rows = []
    for b in sorted(groups):
        rs = groups[b]
        rel = np.array([r["rel_l2_mean"] for r in rs])
        frac = np.array([r["intervention_fraction"] for r in rs])
        rows.append({
            "budget": b,
            "n": len(rs),
            "rel_l2_mean": float(rel.mean()),
            "rel_l2_std": float(rel.std(ddof=1)) if len(rel) > 1 else 0.0,
            "rel_l2_ci95": ci95(rel),
            "intervention_fraction_mean": float(frac.mean()),
        })
    return rows


def main():
    raw_dir = sys.argv[1] if len(sys.argv) > 1 else "results/raw"
    out_dir = "results/processed"
    os.makedirs(out_dir, exist_ok=True)

    recs = load_records(raw_dir)
    print(f"Loaded {len(recs)} records")
    if not recs:
        print("No records found.")
        return

    # Overall summary
    rows = summarize(recs)
    import csv
    with open(os.path.join(out_dir, "summary_by_policy.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print("Wrote", os.path.join(out_dir, "summary_by_policy.csv"))

    # G (BRACE) by budget
    for pol in ["G_brace_budget"]:
        rows_b = summarize_by_budget(recs, pol)
        with open(os.path.join(out_dir, f"{pol}_by_budget.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows_b[0].keys())
            w.writeheader()
            w.writerows(rows_b)
        print("Wrote", os.path.join(out_dir, f"{pol}_by_budget.csv"))

    # Figures
    if plt is not None:
        # Pareto: relL2 vs intervention fraction for each policy
        fig, ax = plt.subplots(figsize=(6, 4))
        for pol in sorted(set(r["policy"] for r in recs)):
            rs = [r for r in recs if r["policy"] == pol]
            frac = np.array([r["intervention_fraction"] for r in rs])
            rel = np.array([r["rel_l2_mean"] for r in rs])
            ax.scatter(frac, rel, label=pol, s=20, alpha=0.7)
        ax.set_xlabel("Intervention fraction")
        ax.set_ylabel("Mean relative L2 error")
        ax.set_title("Accuracy vs intervention cost (all policies)")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "pareto_accuracy_vs_intervention.png"), dpi=150)
        plt.close(fig)

        # G: relL2 vs budget
        rows_b = summarize_by_budget(recs, "G_brace_budget")
        if rows_b:
            fig, ax = plt.subplots(figsize=(5, 4))
            budgets = [r["budget"] for r in rows_b]
            rel = [r["rel_l2_mean"] for r in rows_b]
            ci = [r["rel_l2_ci95"] for r in rows_b]
            ax.errorbar(budgets, rel, yerr=ci, marker="o")
            ax.set_xlabel("Intervention budget")
            ax.set_ylabel("Mean relative L2 error")
            ax.set_title("BRACE-CPU: accuracy vs intervention budget")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, "brace_accuracy_vs_budget.png"), dpi=150)
            plt.close(fig)
        print("Wrote figures to", out_dir)

    # Print headline
    print("\n=== Headline summary ===")
    for r in rows:
        print(f"{r['policy']:<28} relL2={r['rel_l2_mean']:.4f} (+/-{r['rel_l2_ci95']:.4f}) "
              f"frac={r['intervention_fraction_mean']:.3f} solver_ms={r['solver_wall_s_mean']*1000:.1f}")


if __name__ == "__main__":
    main()
