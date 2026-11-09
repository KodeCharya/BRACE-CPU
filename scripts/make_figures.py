"""Generate Pareto and budget figures from corrected rebuild data.

Uses TRUE total cost (including prep for G_TWOPASS) so the figures are honest
about computational cost. Saves to paper/figures/.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    path = sorted(Path("results/rebuild").glob("corrected_*/*_all.json"))[-1]
    records = json.load(open(path))
    out = Path("paper/figures")
    out.mkdir(parents=True, exist_ok=True)

    # Per-policy/budget aggregates
    from collections import defaultdict
    groups = defaultdict(list)
    for r in records:
        groups[(r["policy"], round(r["budget"], 2))].append(r)

    # --- Figure 1: Pareto (true cost vs error) ---
    fig, ax = plt.subplots(figsize=(7, 5))
    policy_styles = {
        "A": ("o", "tab:blue", "A (neural)"),
        "B": ("s", "tab:red", "B (numerical)"),
        "C": ("^", "tab:orange", "C (fixed thresh)"),
        "E": ("D", "tab:purple", "E (uncal. risk)"),
        "F": ("v", "tab:pink", "F (cal. risk)"),
        "G_TWOPASS": ("x", "tab:green", "G_TWOPASS"),
        "G_ONLINE": ("*", "tab:olive", "G_ONLINE"),
    }
    for (pol, bud), recs in sorted(groups.items()):
        rel = np.mean([r["rel_l2_mean"] for r in recs])
        if pol in ("G_TWOPASS", "G_ONLINE"):
            cost = np.mean([r["true_total_ms"] for r in recs])
        else:
            cost = np.mean([r["total_wall_s"] * 1000 for r in recs])
        marker, color, label = policy_styles.get(pol, ("o", "black", pol))
        if pol in ("A", "B"):
            bud_label = "A" if pol == "A" else "B"
            ax.scatter(cost, rel, marker=marker, c=color, s=100, label=bud_label, zorder=5, edgecolors="k")
        else:
            ax.scatter(cost, rel, marker=marker, c=color, s=80, zorder=4, edgecolors="k")
            ax.text(cost + 1, rel, f"{pol.split('_')[0]}/{bud:.0%}", fontsize=6)

    # Add legend entries for policies (first budget point)
    for pol, (marker, color, label) in policy_styles.items():
        ax.scatter([], [], marker=marker, c=color, s=80, label=label, edgecolors="k")

    ax.set_xlabel("True total wall-clock cost (ms, including preparation)")
    ax.set_ylabel("Mean relative L2 error")
    ax.set_title("Accuracy vs true computational cost (corrected, all policies)")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 80)
    fig.tight_layout()
    p1 = out / "pareto_accuracy_vs_true_cost.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    print(f"Saved {p1}")

    # --- Figure 2: G accuracy vs budget (both variants) ---
    fig, ax = plt.subplots(figsize=(6, 4))
    for pol, marker, color, label in [
        ("G_TWOPASS", "x", "tab:green", "G_TWOPASS (two-pass)"),
        ("G_ONLINE", "*", "tab:olive", "G_ONLINE (single-pass)"),
        ("A", "o", "tab:blue", "A (neural)"),
        ("B", "s", "tab:red", "B (numerical)"),
    ]:
        if pol in ("A", "B"):
            recs = groups.get((pol, 1.0 if pol == "B" else 0.0), [])
            if recs:
                ax.axhline(np.mean([r["rel_l2_mean"] for r in recs]), color=color,
                           label=label, linestyle="--", alpha=0.7)
        else:
            budgets, rels, cis = [], [], []
            for bud in [0.05, 0.1, 0.2, 0.4, 0.6]:
                recs = groups.get((pol, bud), [])
                if not recs:
                    continue
                rel = np.array([r["rel_l2_mean"] for r in recs])
                budgets.append(bud)
                rels.append(rel.mean())
                cis.append(1.96 * rel.std(ddof=1) / np.sqrt(len(rel)) if len(rel) > 1 else 0)
            if budgets:
                ax.errorbar(budgets, rels, yerr=cis, marker=marker, color=color,
                            label=label, capsize=3, markersize=10)

    ax.set_xlabel("Intervention budget")
    ax.set_ylabel("Mean relative L2 error")
    ax.set_title("BRACE-G: accuracy vs budget (two-pass vs online)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p2 = out / "brace_accuracy_vs_budget.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    print(f"Saved {p2}")

    # --- Figure 3: Intervention fraction vs nominal budget (budget overshoot) ---
    fig, ax = plt.subplots(figsize=(6, 4))
    for pol, marker, color, label in [
        ("C", "^", "tab:orange", "C (fixed thresh)"),
        ("E", "D", "tab:purple", "E (uncal. risk)"),
        ("F", "v", "tab:pink", "F (cal. risk)"),
        ("G_TWOPASS", "x", "tab:green", "G_TWOPASS"),
        ("G_ONLINE", "*", "tab:olive", "G_ONLINE"),
    ]:
        budgets, fracs = [], []
        for bud in [0.05, 0.1, 0.2, 0.4, 0.6]:
            recs = groups.get((pol, bud), [])
            if not recs:
                continue
            budgets.append(bud)
            fracs.append(np.mean([r["intervention_fraction"] for r in recs]))
        if budgets:
            ax.plot(budgets, fracs, marker=marker, color=color, label=label, linestyle="-")

    ax.plot([0, 0.65], [0, 0.65], "k--", alpha=0.5, label="Nominal budget (diagonal)")
    ax.set_xlabel("Nominal budget")
    ax.set_ylabel("Actual intervention fraction")
    ax.set_title("Budget compliance: actual vs nominal intervention fraction")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 0.65)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    p3 = out / "budget_compliance.png"
    fig.savefig(p3, dpi=150)
    plt.close(fig)
    print(f"Saved {p3}")


if __name__ == "__main__":
    main()
