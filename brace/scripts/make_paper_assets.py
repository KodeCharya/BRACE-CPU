"""Convert result CSVs into LaTeX tables for the manuscript.

All numbers are read from ``results/tables``; nothing is hard-coded. Missing
cells are printed as ``--`` so an incomplete matrix is visible rather than
silently filled.
"""
from __future__ import annotations

import csv
import os

SPLITS = [
    ("burgers1d", "test"), ("burgers1d", "ood_param"), ("burgers1d", "ood_ic"),
    ("allencahn1d", "test"), ("allencahn1d", "ood_param"), ("allencahn1d", "ood_ic"),
    ("advection2d", "test"), ("advection2d", "ood_param"), ("advection2d", "ood_ic"),
]
MAIN_POLICIES = ["neural", "fixed", "adaptive", "uniform", "early",
                 "ranked_residual", "ranked_ensemble", "ranked_calibrated",
                 "brace", "brace_combo", "gain_ranked", "oracle_gain",
                 "oracle", "reanchor"]
BUDGETS = ["0.1", "0.2", "0.4"]


def read_csv(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def fmt(v: str, nd: int = 3) -> str:
    try:
        return f"{float(v):.{nd}f}"
    except (ValueError, TypeError):
        return v


def fmt_budget(b: str) -> str:
    return f"{float(b):g}" if b not in ("", None) else "--"


def policy_label(p: str) -> str:
    return {"ranked_calibrated": "calib.-ranked", "ranked_ensemble": "ensemble-ranked",
            "ranked_residual": "residual-ranked", "brace_combo": "BRACE-fusion",
            "brace": "BRACE", "gain_ranked": "gain-ranked",
            "oracle_gain": "gain oracle"}.get(p, p)


def _lookup(rows, bench, split, policy, budget):
    for r in rows:
        if (r["benchmark"] == bench and r["split"] == split and r["policy"] == policy
                and r.get("variant", "") == "" and r["budget"] == budget):
            return r
    return None


def budget_table(exp_id: str = "main") -> str:
    """One row per policy; error/cost at each declared budget."""
    rows = read_csv(f"results/tables/{exp_id}_at_budget.csv")
    agg = read_csv(f"results/tables/{exp_id}_aggregate.csv")
    lines = [r"\begin{tabular}{lllrrr}", r"\toprule",
             r"Benchmark & Split & Policy & $B=0.1$ & $B=0.2$ & $B=0.4$ \\",
             r"\midrule"]
    for bench, split in SPLITS:
        any_row = False
        base = [r for r in agg if r["benchmark"] == bench and r["split"] == split
                and r["policy"] == "neural"]
        if base:
            cell = f"{fmt(base[0]['rel_l2_mean'])}/0.00"
            lines.append(f"{bench} & {split} & surrogate only & "
                         + " & ".join([cell] * len(BUDGETS)) + r" \\")
            any_row = True
        for pol in MAIN_POLICIES:
            if pol == "neural":
                continue
            cells = []
            for b in BUDGETS:
                r = _lookup(rows, bench, split, pol, b)
                cells.append(f"{fmt(r['rel_l2_mean'])}/{fmt(r['cost_fraction'], 2)}"
                             if r else "--")
            if any(c != "--" for c in cells):
                any_row = True
                lines.append(f"{bench} & {split} & {policy_label(pol)} & "
                             + " & ".join(cells) + r" \\")
        if any_row:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def main_table(exp_id: str = "main") -> str:
    """Long-form version, convenient for supplementary material."""
    rows = read_csv(f"results/tables/{exp_id}_aggregate.csv")
    lines = [
        r"\begin{tabular}{llllrrrr}",
        r"\toprule",
        r"Benchmark & Split & Policy & $B$ & rel.\ $L_2$ & 95\% CI & cost & failure \\",
        r"\midrule",
    ]
    for bench, split in SPLITS:
        sub = [r for r in rows if r["benchmark"] == bench and r["split"] == split
               and r.get("variant", "") == ""]
        for r in sub:
            b = r["budget"] if r["budget"] else "--"
            lines.append(
                f"{bench} & {split} & {r['policy']} & {b} & "
                f"{fmt(r['rel_l2_mean'])} & {fmt(r['rel_l2_ci95'])} & "
                f"{fmt(r['cost_fraction'],2)} & {fmt(r['step_failure_rate'],3)} \\\\"
            )
        lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def risk_table(exp_id: str = "main") -> str:
    rows = read_csv(f"results/tables/{exp_id}_risk.csv")
    lines = [r"\begin{tabular}{llrrrrrr}", r"\toprule",
             r"Benchmark & Split & AUROC res. & AUROC ens. & AUROC cal. & ECE & Brier & cov. \\",
             r"\midrule"]
    for bench, split in SPLITS:
        sub = [r for r in rows if r["benchmark"] == bench and r["split"] == split]
        for r in sub:
            lines.append(
                f"{bench} & {split} & {fmt(r['auroc_residual'])} & "
                f"{fmt(r['auroc_ensemble'])} & {fmt(r['auroc_calibrated'])} & "
                f"{fmt(r['ece'])} & {fmt(r['brier'])} & "
                f"{fmt(r['conformal_coverage'])} \\\\"
            )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def paired_table(exp_id: str = "main") -> str:
    rows = read_csv(f"results/tables/{exp_id}_paired.csv")
    keys = {"brace vs neural", "brace vs fixed", "brace vs adaptive",
            "brace vs ranked_ensemble", "brace vs ranked_calibrated",
            "brace_combo vs brace", "brace_combo vs ranked_ensemble",
            "brace_combo vs early", "ranked_ensemble vs neural"}
    lines = [r"\begin{tabular}{llllrrrr}", r"\toprule",
             r"Benchmark & Split & $B$ & Comparison & $\Delta$ & 95\% CI & $d_z$ & sig. \\",
             r"\midrule"]
    for bench, split in SPLITS:
        for r in rows:
            if r["benchmark"] != bench or r["split"] != split or r.get("variant", ""):
                continue
            comp = r["comparison"]
            comp_norm = comp.replace(" vs ", " vs ")
            if comp_norm not in keys and comp not in keys:
                continue
            lines.append(
                f"{bench} & {split} & {r['budget'] or '--'} & "
                f"{comp} & {fmt(r['mean_diff'])} & "
                f"{fmt(r['ci95_across_seeds'])} & {fmt(r['cohen_dz_mean'],2)} & "
                f"{r['seeds_sig_005']}/{r['n_seeds']} \\\\"
            )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def ablation_table(exp_id: str = "ablation") -> str:
    rows = read_csv(f"results/tables/{exp_id}_at_budget.csv") or \
        read_csv(f"results/tables/{exp_id}_aggregate.csv")
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Variant & $B$ & rel.\ $L_2$ & cost & failure \\", r"\midrule"]
    for r in rows:
        if r["policy"] != "brace":
            continue
        lines.append(f"{r.get('variant','') or 'full'} & {r['budget']} & "
                     f"{fmt(r['rel_l2_mean'])} & {fmt(r['cost_fraction'],2)} & "
                     f"{fmt(r['step_failure_rate'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def deeponet_table(exp_id: str = "deeponet") -> str:
    rows = read_csv(f"results/tables/{exp_id}_at_budget.csv")
    agg = read_csv(f"results/tables/{exp_id}_aggregate.csv")
    lines = [r"\begin{tabular}{llrrrr}", r"\toprule",
             r"Policy & $B$ & median rel.\ $L_2$ & mean rel.\ $L_2$ & cost & diverged \\",
             r"\midrule"]
    base = [r for r in agg if r["benchmark"] == "burgers1d" and r["split"] == "test"
            and r["policy"] == "neural"]
    if base:
        lines.append(f"surrogate only & -- & {fmt(base[0]['rel_l2_median'])} & "
                     f"{fmt(base[0]['rel_l2_mean'])} & 0.00 & "
                     f"{fmt(base[0]['divergence_rate'],2)} \\\\")
    for pol in ["ranked_ensemble", "brace"]:
        for b in BUDGETS:
            r = _lookup(rows, "burgers1d", "test", pol, b)
            if r:
                lines.append(f"{policy_label(pol)} & {b} & {fmt(r['rel_l2_median'])} & "
                             f"{fmt(r['rel_l2_mean'])} & {fmt(r['cost_fraction'],2)} & "
                             f"{fmt(r['divergence_rate'],2)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def main() -> None:
    os.makedirs("paper/tables", exist_ok=True)
    os.makedirs("paper/figures", exist_ok=True)
    for name, fn in [("main_results", budget_table), ("main_long", main_table),
                     ("risk", risk_table), ("paired", paired_table),
                     ("ablation", ablation_table), ("deeponet", deeponet_table)]:
        tex = fn()
        if tex:
            with open(f"paper/tables/{name}.tex", "w") as f:
                f.write(tex + "\n")
            print(f"wrote paper/tables/{name}.tex")


if __name__ == "__main__":
    main()
