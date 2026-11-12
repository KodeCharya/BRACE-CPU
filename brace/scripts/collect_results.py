"""Collect the numbers used for reporting into one auditable digest.

Reads the processed CSVs produced by ``scripts/analyze.py`` and writes both a
JSON file and a plain-text summary. Nothing is hard-coded: every value in the
digest can be traced back to ``results/tables`` or ``results/raw``.
"""
from __future__ import annotations

import csv
import json
import os

OUT_JSON = "results/processed/results_digest.json"
OUT_TXT = "results/processed/results_digest.txt"


def read_csv(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def main() -> None:
    main_rows = read_csv("results/tables/main_at_budget.csv")
    risk_rows = read_csv("results/tables/main_risk.csv")
    paired = read_csv("results/tables/main_paired.csv")
    abl = read_csv("results/tables/ablation_at_budget.csv")
    corr = read_csv("results/tables/correction_at_budget.csv")
    gain = read_csv("results/tables/main_correction_gain.csv")
    deep = read_csv("results/tables/deeponet_at_budget.csv")

    digest = {"main": {}, "best_at_budget": {}, "budget_fidelity": {},
              "risk": {}, "paired": {}, "ablation": {}, "correction_mode": {},
              "correction_gain": {}, "deeponet": {}}

    # Main table, keyed by benchmark/split/policy/budget.
    for r in main_rows:
        key = (r["benchmark"], r["split"], r["policy"], r["budget"])
        digest["main"]["|".join(key)] = {k: fnum(r.get(k)) for k in
                                         ("rel_l2_mean", "rel_l2_median", "rel_l2_ci95",
                                          "cost_fraction", "step_failure_rate",
                                          "divergence_rate")}

    # Best risk-aware policy per benchmark/split/budget (excluding oracle).
    groups = {}
    for r in main_rows:
        if r["policy"] in ("neural", "numerical", "oracle"):
            continue
        if fnum(r["cost_fraction"]) is None:
            continue
        g = (r["benchmark"], r["split"], r["budget"])
        groups.setdefault(g, []).append(r)
    for g, rows in sorted(groups.items()):
        best = min(rows, key=lambda r: fnum(r["rel_l2_mean"]))
        digest["best_at_budget"]["|".join(g)] = {
            "policy": best["policy"], "rel_l2": fnum(best["rel_l2_mean"]),
            "cost": fnum(best["cost_fraction"])}

    # Declared vs realised cost (the fidelity claim).
    for r in main_rows:
        c, b = fnum(r["cost_fraction"]), fnum(r["budget"])
        if c is None or b is None:
            continue
        digest["budget_fidelity"][f"{r['benchmark']}|{r['split']}|{r['policy']}|{r['budget']}"] = {
            "declared": b, "realised": c, "delta": c - b}

    for r in risk_rows:
        digest["risk"][f"{r['benchmark']}|{r['split']}|{r['seed']}"] = {
            k: fnum(r.get(k)) for k in
            ("file", "failure_rate", "auroc_residual", "auroc_ensemble",
             "auroc_calibrated", "auroc_brace", "ece", "brier",
             "conformal_coverage", "conformal_upper_coverage")}

    for r in paired:
        digest["paired"][f"{r['benchmark']}|{r['split']}|{r['budget']}|{r['comparison']}"] = {
            k: fnum(r.get(k)) for k in
            ("mean_diff", "ci95_across_seeds", "cohen_dz_mean", "p_min", "p_max",
             "seeds_sig_005", "n_seeds")}

    for r in abl:
        digest["ablation"][f"{r.get('variant','')}|{r['policy']}|{r['budget']}"] = {
            "rel_l2": fnum(r["rel_l2_mean"]), "cost": fnum(r["cost_fraction"])}

    for r in corr:
        digest["correction_mode"][f"{r['policy']}|{r['budget']}"] = {
            "rel_l2": fnum(r["rel_l2_mean"]), "cost": fnum(r["cost_fraction"])}

    for r in gain:
        digest["correction_gain"][f"{r['benchmark']}|{r['policy']}|{r['bin']}"] = {
            "gain": fnum(r.get("mean_gain")), "action_rate": fnum(r.get("action_rate"))}

    for r in deep:
        digest["deeponet"][f"{r['policy']}|{r['budget']}"] = {
            "rel_l2_mean": fnum(r["rel_l2_mean"]), "rel_l2_median": fnum(r["rel_l2_median"]),
            "cost": fnum(r["cost_fraction"]), "divergence": fnum(r["divergence_rate"])}

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(digest, f, indent=2)

    lines = ["BRACE results digest", "====================", ""]
    lines.append("Main results (rel-L2 / realised cost):")
    for r in main_rows:
        lines.append(f"  {r['benchmark']:12s} {r['split']:9s} {r['policy']:18s} "
                     f"B={r['budget']:>4s}  {fnum(r['rel_l2_mean']):.4f} / "
                     f"{fnum(r['cost_fraction']):.3f}")
    lines.append("")
    lines.append("Risk (in-distribution and shifted):")
    for r in risk_rows:
        lines.append(f"  {r['benchmark']:12s} {r['split']:9s} s{r['seed']}  "
                     f"AUROC cal/res/ens = {fnum(r['auroc_calibrated']):.3f}/"
                     f"{fnum(r['auroc_residual']):.3f}/{fnum(r['auroc_ensemble']):.3f}  "
                     f"ECE={fnum(r['ece']):.3f}  cov={fnum(r['conformal_coverage']):.3f}")
    with open(OUT_TXT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[:60]))
    print(f"\nwrote {OUT_JSON} and {OUT_TXT}")


if __name__ == "__main__":
    main()
