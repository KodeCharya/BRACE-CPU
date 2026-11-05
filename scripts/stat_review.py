"""Clean statistical reanalysis from corrected raw records.
Verifies all manuscript claims and produces correct paired comparisons.
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


def get_subset(records, policy, budget=None):
    """Return rel_l2_mean array for records matching policy (+ optional budget)."""
    vals = []
    for r in records:
        if r["policy"] != policy:
            continue
        if budget is not None and abs(r["budget"] - budget) > 1e-9:
            continue
        vals.append(r["rel_l2_mean"])
    return np.array(vals)


def paired_compare(records, polA, budA, polB, budB):
    """Paired diff = A - B on (seed, trajectory) pairs. Both must exist for a pair."""
    by_st = defaultdict(dict)
    for r in records:
        st = (r["seed"], r["test_trajectory_id"])
        key = (r["policy"], round(r["budget"], 2))
        by_st[st][key] = r["rel_l2_mean"]
    ka = (polA, round(budA, 2)) if budA is not None else (polA, None)
    kb = (polB, round(budB, 2)) if budB is not None else (polB, None)
    diffs = []
    for st, d in by_st.items():
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
    print()

    # --- Per-policy/budget actual performance ---
    print("=" * 110)
    print("ACTUAL PERFORMANCE BY (POLICY, NOMINAL BUDGET)")
    print("=" * 110)
    print(f"{'Policy':<28} {'nomBudg':>8} {'n':>5} {'relL2 mean':>12} {'CI95':>10} "
          f"{'actualFrac':>11} {'total_ms':>9}")
    for pol in ["A_always_neural", "B_always_numerical", "H_oracle",
                "C_fixed_threshold", "E_uncalibrated_risk", "F_calibrated_risk",
                "G_brace_budget", "D_adaptive_threshold"]:
        budgets = [None] if pol in ("A_always_neural", "B_always_numerical", "H_oracle") \
            else [0.05, 0.1, 0.2, 0.4, 0.6]
        for bud in budgets:
            vals = get_subset(records, pol, bud)
            if len(vals) == 0:
                continue
            # actual frac from records
            fracs = [r["intervention_fraction"] for r in records
                     if r["policy"] == pol and (bud is None or abs(r["budget"] - bud) < 1e-9)]
            tots = [r["total_wall_s"] * 1000 for r in records
                    if r["policy"] == pol and (bud is None or abs(r["budget"] - bud) < 1e-9)]
            rel_mean = float(vals.mean())
            rel_ci = ci95(vals)
            frac_mean = float(np.mean(fracs)) if fracs else 0.0
            tot_mean = float(np.mean(tots)) if tots else 0.0
            bud_label = f"{bud:.2f}" if bud is not None else "---"
            print(f"{pol:<28} {bud_label:>8} {len(vals):>5} {rel_mean:>12.5f} "
                  f"{rel_ci:>10.5f} {frac_mean:>11.3f} {tot_mean:>9.2f}")
    print()

    # --- Manuscript table verification ---
    print("=" * 110)
    print("MANUSCRIPT TABLE VERIFICATION")
    print("=" * 110)
    # Manuscript claims (from tab:summary)
    claims = [
        ("A always neural", None, 0.2550, 0.0207, 0.000, 8.50),
        ("B always numerical", None, 0.0000, 0.0000, 1.000, 10.98),
        ("H oracle", None, 0.0042, 0.0004, 0.955, 21.70),
        ("G BRACE", 0.05, 0.2217, 0.0175, 0.050, 8.67),
        ("G BRACE", 0.10, 0.1970, 0.0162, 0.100, 8.82),
        ("G BRACE", 0.20, 0.1588, 0.0152, 0.200, 9.09),
        ("G BRACE", 0.40, 0.1083, 0.0128, 0.400, 9.63),
        ("G BRACE", 0.60, 0.0645, 0.0088, 0.600, 10.15),
        # C/E/F at nominal 0.05 — manuscript claims these, but need to check what it actually means
        ("C fixed threshold", 0.05, 0.0490, 0.0049, 0.660, 20.87),
        ("E uncalibrated risk", 0.05, 0.0489, 0.0051, 0.658, 56.07),
        ("F calibrated risk", 0.05, 0.0489, 0.0051, 0.658, 66.89),
    ]
    base = float(get_subset(records, "A_always_neural").mean())
    all_ok = True
    for pol, bud, rel_m, ci_m, frac_m, tot_m in claims:
        vals = get_subset(records, pol, bud)
        if len(vals) == 0:
            print(f"{pol} bud={bud}: NO RECORDS")
            all_ok = False
            continue
        rel_mean = float(vals.mean()); rel_ci = ci95(vals)
        fr = [r["intervention_fraction"] for r in records
              if r["policy"] == pol and (bud is None or abs(r["budget"] - bud) < 1e-9)]
        tt = [r["total_wall_s"] * 1000 for r in records
              if r["policy"] == pol and (bud is None or abs(r["budget"] - bud) < 1e-9)]
        frac_mean = float(np.mean(fr)); tot_mean = float(np.mean(tt))
        rel_ok = abs(rel_mean - rel_m) < 0.0005
        ci_ok = abs(rel_ci - ci_m) < 0.0005
        frac_ok = abs(frac_mean - frac_m) < 0.005
        tot_ok = abs(tot_mean - tot_m) < 0.05
        ok = rel_ok and ci_ok and frac_ok and tot_ok
        all_ok = all_ok and ok
        red = (base - rel_mean) / base * 100
        print(f"{pol:<28} bud={str(bud):>4} relL2={rel_mean:.5f}(claim={rel_m:.4f}) "
              f"{'OK' if rel_ok else 'MISMATCH'} | CI={rel_ci:.5f}(claim={ci_m:.4f}) "
              f"{'OK' if ci_ok else 'MISMATCH'} | frac={frac_mean:.3f}(claim={frac_m:.3f}) "
              f"{'OK' if frac_ok else 'MISMATCH'} | tot={tot_mean:.2f}(claim={tot_m:.2f}) "
              f"{'OK' if tot_ok else 'MISMATCH'} | reduction={red:.1f}%")
    print(f"ALL CLAIMS MATCH: {all_ok}")
    print()

    # --- Paired comparisons ---
    print("=" * 110)
    print("PAIRED COMPARISONS (diff = A - B; negative = B more accurate; n=125 seed-traj pairs)")
    print("=" * 110)
    comps = [
        ("G_brace_budget", 0.2, "A_always_neural", None, "G(0.2) - A(neural)"),
        ("G_brace_budget", 0.4, "A_always_neural", None, "G(0.4) - A(neural)"),
        ("G_brace_budget", 0.6, "A_always_neural", None, "G(0.6) - A(neural)"),
        ("C_fixed_threshold", 0.05, "G_brace_budget", 0.2,
         "C(nom0.05,act~0.20) - G(0.2)   [matched actual frac]"),
        ("C_fixed_threshold", 0.05, "C_fixed_threshold", 0.2,
         "C(nom0.05) - C(nom0.2)   [same policy, diff budgets]"),
        ("H_oracle", None, "G_brace_budget", 0.2, "H(oracle) - G(0.2)"),
        ("F_calibrated_risk", 0.05, "C_fixed_threshold", 0.05,
         "F(cal,nom0.05) - C(fixed,nom0.05)   [calibration null]"),
        ("E_uncalibrated_risk", 0.05, "F_calibrated_risk", 0.05,
         "E(uncal,nom0.05) - F(cal,nom0.05)   [calibration adds nothing]"),
    ]
    for polA, budA, polB, budB, label in comps:
        res = paired_compare(records, polA, budA, polB, budB)
        if res is None:
            print(f"{label:<62} NO DATA")
            continue
        sig = "***" if res["pval"] < 0.001 else ("**" if res["pval"] < 0.01
                                                  else ("*" if res["pval"] < 0.05 else "ns"))
        print(f"{label:<62} n={res['n']:3d} diff={res['mean_diff']:+.5f} "
              f"CI95=[{res['ci_lo']:+.5f},{res['ci_hi']:+.5f}] "
              f"d={res['cohen']:+.3f} p={res['pval']:.2e} {sig}")
    print()
    print("NOTE: negative diff = second policy (B) is more accurate (lower error).")
    print("      Negative Cohen's d = B better than A.")


if __name__ == "__main__":
    main()
