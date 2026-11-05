"""Evidence traceability audit: recalculate all manuscript statistics from
corrected raw records and verify against manuscript claims.

Run: PYTHONPATH=src .venv/bin/python scripts/evidence_audit.py
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


RAW = Path("results/raw/exp_20261105_134910_63a5fa94_all.json")


def ci95(x):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return 0.0
    return float(np.std(x, ddof=1) / np.sqrt(len(x)) * 1.96)


def load_records():
    with open(RAW) as f:
        return json.load(f)


def summarize_by_policy(records):
    groups = defaultdict(list)
    for r in records:
        groups[r["policy"]].append(r)
    out = {}
    for pol, rs in groups.items():
        rel = np.array([r["rel_l2_mean"] for r in rs])
        frac = np.array([r["intervention_fraction"] for r in rs])
        neu = np.array([r["neural_wall_s"] for r in rs]) * 1000
        sol = np.array([r["solver_wall_s"] for r in rs]) * 1000
        ctrl = np.array([r["controller_wall_s"] for r in rs]) * 1000
        prep = np.array([r["prepare_wall_s"] for r in rs]) * 1000
        tot = np.array([r["total_wall_s"] for r in rs]) * 1000
        out[pol] = dict(
            n=len(rs),
            rel_mean=float(rel.mean()),
            rel_std=float(rel.std(ddof=1)) if len(rel) > 1 else 0.0,
            rel_ci95=ci95(rel),
            frac_mean=float(frac.mean()),
            frac_std=float(frac.std(ddof=1)) if len(frac) > 1 else 0.0,
            neu_mean=float(neu.mean()),
            sol_mean=float(sol.mean()),
            ctrl_mean=float(ctrl.mean()),
            prep_mean=float(prep.mean()),
            tot_mean=float(tot.mean()),
        )
    return out


def paired_compare(records, polA, budA, polB, budB):
    """Paired comparison on (seed, trajectory) pairs."""
    by_st = defaultdict(dict)
    for r in records:
        key = (polA, round(budA, 2)) if budA is not None else polA
        by_st[(r["seed"], r["test_trajectory_id"])][key] = r["rel_l2_mean"]
        key2 = (polB, round(budB, 2)) if budB is not None else polB
        by_st[(r["seed"], r["test_trajectory_id"])][key2] = r["rel_l2_mean"]
    diffs = []
    for (s, t), d in by_st.items():
        ka = (polA, round(budA, 2)) if budA is not None else polA
        kb = (polB, round(budB, 2)) if budB is not None else polB
        if ka in d and kb in d:
            diffs.append(d[ka] - d[kb])
    diffs = np.array(diffs)
    if len(diffs) == 0:
        return None
    n = len(diffs)
    md = float(diffs.mean())
    sd = float(diffs.std(ddof=1)) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 0 else 0.0
    ci_lo = md - 1.96 * se
    ci_hi = md + 1.96 * se
    cohen = md / sd if sd > 0 else 0.0
    tstat = md / se if se > 0 else 0.0
    pval = float(2 * (1 - stats.t.cdf(abs(tstat), df=n - 1))) if n > 1 else 1.0
    return dict(n=n, mean_diff=md, sd=sd, se=se, ci_lo=ci_lo, ci_hi=ci_hi,
                cohen=cohen, tstat=tstat, pval=pval)


def main():
    records = load_records()
    print(f"Loaded {len(records)} corrected records")
    print(f"Seeds: {sorted(set(r['seed'] for r in records))}")
    print(f"Budgets: {sorted(set(r['budget'] for r in records))}")
    print(f"Policies: {sorted(set(r['policy'] for r in records))}")
    print()

    summary = summarize_by_policy(records)
    print("=" * 100)
    print("RECALCULATED SUMMARY (corrected raw records)")
    print("=" * 100)
    print(f"{'Policy':<26} {'n':>4} {'relL2 mean':>11} {'std':>9} {'CI95':>9} "
          f"{'frac':>7} {'neural_ms':>9} {'solver_ms':>9} {'ctrl_ms':>9} "
          f"{'prep_ms':>8} {'total_ms':>9}")
    for pol in sorted(summary):
        s = summary[pol]
        print(f"{pol:<26} {s['n']:>4} {s['rel_mean']:>11.5f} {s['rel_std']:>9.5f} "
              f"{s['rel_ci95']:>9.5f} {s['frac_mean']:>7.3f} "
              f"{s['neu_mean']:>9.2f} {s['sol_mean']:>9.2f} {s['ctrl_mean']:>9.2f} "
              f"{s['prep_mean']:>8.2f} {s['tot_mean']:>9.2f}")

    print()
    print("=" * 100)
    print("PAIRED COMPARISONS (seed-trajectory clustered, n=125 pairs)")
    print("=" * 100)
    comparisons = [
        ("G_brace_budget", 0.2, "A_always_neural", None, "G(0.2) vs A(neural)"),
        ("G_brace_budget", 0.4, "A_always_neural", None, "G(0.4) vs A(neural)"),
        ("G_brace_budget", 0.6, "A_always_neural", None, "G(0.6) vs A(neural)"),
        ("C_fixed_threshold", 0.05, "G_brace_budget", 0.2,
         "C(0.05nom, act~0.66) vs G(0.2)"),
        ("H_oracle", None, "G_brace_budget", 0.2, "H(oracle) vs G(0.2)"),
        ("F_calibrated_risk", 0.05, "C_fixed_threshold", 0.05,
         "F(cal, 0.05nom) vs C(fixed, 0.05nom)"),
    ]
    for polA, budA, polB, budB, label in comparisons:
        res = paired_compare(records, polA, budA, polB, budB)
        if res is None:
            print(f"{label:<55} no data")
            continue
        sig = "***" if res["pval"] < 0.001 else ("**" if res["pval"] < 0.01
                                                  else ("*" if res["pval"] < 0.05 else "ns"))
        print(f"{label:<55} n={res['n']:3d} diff={res['mean_diff']:+.5f} "
              f"CI95=[{res['ci_lo']:+.5f},{res['ci_hi']:+.5f}] "
              f"d={res['cohen']:+.3f} p={res['pval']:.2e} {sig}")

    print()
    print("=" * 100)
    print("MANUSCRIPT CLAIM VERIFICATION (corrected raw vs manuscript table)")
    print("=" * 100)
    manuscript = {
        "A always neural": (0.2550, 0.0207, 0.000, 8.50),
        "B always numerical": (0.0000, 0.0000, 1.000, 10.98),
        "H oracle": (0.0042, 0.0004, 0.955, 21.70),
        "G BRACE 0.05": (0.2217, 0.0175, 0.050, 8.67),
        "G BRACE 0.10": (0.1970, 0.0162, 0.100, 8.82),
        "G BRACE 0.20": (0.1588, 0.0152, 0.200, 9.09),
        "G BRACE 0.40": (0.1083, 0.0128, 0.400, 9.63),
        "G BRACE 0.60": (0.0645, 0.0088, 0.600, 10.15),
        "C fixed threshold 0.05": (0.0490, 0.0049, 0.660, 20.87),
        "E uncalibrated risk 0.05": (0.0489, 0.0051, 0.658, 56.07),
        "F calibrated risk 0.05": (0.0489, 0.0051, 0.658, 66.89),
    }
    # map manuscript key -> (policy, budget)
    key_map = {
        "A always neural": ("A_always_neural", None),
        "B always numerical": ("B_always_numerical", None),
        "H oracle": ("H_oracle", None),
        "G BRACE 0.05": ("G_brace_budget", 0.05),
        "G BRACE 0.10": ("G_brace_budget", 0.10),
        "G BRACE 0.20": ("G_brace_budget", 0.20),
        "G BRACE 0.40": ("G_brace_budget", 0.40),
        "G BRACE 0.60": ("G_brace_budget", 0.60),
        "C fixed threshold 0.05": ("C_fixed_threshold", 0.05),
        "E uncalibrated risk 0.05": ("E_uncalibrated_risk", 0.05),
        "F calibrated risk 0.05": ("F_calibrated_risk", 0.05),
    }
    all_ok = True
    for mkey, (rel_m, ci_m, frac_m, tot_m) in manuscript.items():
        pol, bud = key_map[mkey]
        s = summary.get(pol)
        if s is None:
            print(f"{mkey:<32} POLICY NOT FOUND")
            all_ok = False
            continue
        if bud is not None:
            # filter by budget
            sub = [r for r in records if r["policy"] == pol and abs(r["budget"] - bud) < 1e-9]
            rel = np.array([x["rel_l2_mean"] for x in sub])
            frac = np.array([x["intervention_fraction"] for x in sub])
            tot = np.array([x["total_wall_s"] for x in sub]) * 1000
            rel_mean = float(rel.mean())
            rel_ci = ci95(rel)
            frac_mean = float(frac.mean())
            tot_mean = float(tot.mean())
        else:
            rel_mean = s["rel_mean"]; rel_ci = s["rel_ci95"]
            frac_mean = s["frac_mean"]; tot_mean = s["tot_mean"]
        rel_ok = abs(rel_mean - rel_m) < 0.0005
        ci_ok = abs(rel_ci - ci_m) < 0.0005
        frac_ok = abs(frac_mean - frac_m) < 0.005
        tot_ok = abs(tot_mean - tot_m) < 0.05
        ok = rel_ok and ci_ok and frac_ok and tot_ok
        all_ok = all_ok and ok
        print(f"{mkey:<32} relL2={rel_mean:.5f}(ms={rel_m:.4f}) {'OK' if rel_ok else 'MISMATCH'} | "
              f"CI95={rel_ci:.5f}(ms={ci_m:.4f}) {'OK' if ci_ok else 'MISMATCH'} | "
              f"frac={frac_mean:.3f}(ms={frac_m:.3f}) {'OK' if frac_ok else 'MISMATCH'} | "
              f"tot_ms={tot_mean:.2f}(ms={tot_m:.2f}) {'OK' if tot_ok else 'MISMATCH'}")
    print()
    print("ALL MANUSCRIPT TABLE VALUES VERIFIED:" , all_ok)

    # G per-budget error reduction vs neural
    print()
    print("=" * 100)
    print("G ERROR REDUCTION vs A (neural baseline 0.2550)")
    print("=" * 100)
    base = summary["A_always_neural"]["rel_mean"]
    for bud in [0.05, 0.1, 0.2, 0.4, 0.6]:
        sub = [r for r in records if r["policy"] == "G_brace_budget" and abs(r["budget"] - bud) < 1e-9]
        rel = np.array([x["rel_l2_mean"] for x in sub])
        red = (base - rel.mean()) / base * 100
        print(f"budget={bud}: relL2={rel.mean():.5f} reduction_vs_neural={red:.1f}%")

    # G per-budget CI95 for figure
    print()
    print("G per-budget CI95 (for brace_accuracy_vs_budget figure):")
    for bud in [0.05, 0.1, 0.2, 0.4, 0.6]:
        sub = [r for r in records if r["policy"] == "G_brace_budget" and abs(r["budget"] - bud) < 1e-9]
        rel = np.array([x["rel_l2_mean"] for x in sub])
        print(f"budget={bud}: mean={rel.mean():.5f} ci95={ci95(rel):.5f}")


if __name__ == "__main__":
    main()
