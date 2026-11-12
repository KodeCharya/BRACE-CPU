"""Convert result CSVs into LaTeX tables for the manuscript."""
from __future__ import annotations

import csv
import os


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


def main_table(exp_id: str = "main") -> str:
    rows = read_csv(f"results/tables/{exp_id}_aggregate.csv")
    keep = [("burgers1d", "test"), ("burgers1d", "ood_param"), ("burgers1d", "ood_ic"),
            ("allencahn1d", "test"), ("advection2d", "test")]
    lines = [
        r"\begin{tabular}{llllrrrr}",
        r"\toprule",
        r"Benchmark & Split & Policy & $B$ & rel.\ $L_2$ & 95\% CI & cost & failure \\",
        r"\midrule",
    ]
    for bench, split in keep:
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
    for r in rows:
        lines.append(
            f"{r['benchmark']} & {r['split']} & {fmt(r['auroc_residual'])} & "
            f"{fmt(r['auroc_ensemble'])} & {fmt(r['auroc_calibrated'])} & "
            f"{fmt(r['ece'])} & {fmt(r['brier'])} & {fmt(r['conformal_coverage'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def paired_table(exp_id: str = "main") -> str:
    rows = read_csv(f"results/tables/{exp_id}_paired.csv")
    lines = [r"\begin{tabular}{llllrrr}", r"\toprule",
             r"Benchmark & Split & $B$ & Comparison & $\Delta$ & 95\% CI & $d_z$ \\",
             r"\midrule"]
    for r in rows:
        if not r.get("variant", ""):
            lines.append(
                f"{r['benchmark']} & {r['split']} & {r['budget'] or '--'} & "
                f"{r['comparison']} & {fmt(r['mean_diff'])} & "
                f"{fmt(r['ci95_across_seeds'])} & {fmt(r['cohen_dz_mean'],2)} \\\\"
            )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def ablation_table(exp_id: str = "ablation") -> str:
    rows = read_csv(f"results/tables/{exp_id}_aggregate.csv")
    lines = [r"\begin{tabular}{llrrr}", r"\toprule",
             r"Variant & $B$ & rel.\ $L_2$ & cost & failure \\", r"\midrule"]
    for r in rows:
        if r["policy"] != "brace":
            continue
        lines.append(f"{r.get('variant','') or 'full'} & {r['budget']} & "
                     f"{fmt(r['rel_l2_mean'])} & {fmt(r['cost_fraction'],2)} & "
                     f"{fmt(r['step_failure_rate'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def main() -> None:
    os.makedirs("paper/tables", exist_ok=True)
    os.makedirs("paper/figures", exist_ok=True)
    for name, fn in [("main_results", main_table), ("risk", risk_table),
                     ("paired", paired_table), ("ablation", ablation_table)]:
        tex = fn()
        if tex:
            with open(f"paper/tables/{name}.tex", "w") as f:
                f.write(tex + "\n")
            print(f"wrote paper/tables/{name}.tex")


if __name__ == "__main__":
    main()
