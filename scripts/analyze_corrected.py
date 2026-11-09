"""Analyze corrected experiment results with complete cost accounting.

Reads the corrected experiment JSON and produces:
1. Per-(policy, budget) summary with TRUE total cost (including prep)
2. Paired statistical comparisons at pair, seed, and trajectory levels
3. Accuracy-vs-cost analysis at matched intervention fraction
4. Budget enforcement verification
5. Comparison of G_TWOPASS vs G_ONLINE
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


def ci95(x):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return 0.0
    return float(np.std(x, ddof=1) / np.sqrt(len(x)) * 1.96)


def paired(diffs):
    n = len(diffs)
    md = float(np.mean(diffs))
    sd = float(np.std(diffs, ddof=1)) if n > 1 else 0.0
    se = sd / math.sqrt(n)
    t = md / se if se > 0 else 0.0
    p = float(2 * (1 - stats.t.cdf(abs(t), df=n - 1))) if n > 1 else 1.0
    cohen = md / sd if sd > 0 else 0.0
    return n, md, sd, md - 1.96 * se, md + 1.96 * se, t, p, cohen


def main():
    path = sorted(Path("results/rebuild").glob("corrected_*/*_all.json"))[-1]
    records = json.load(open(path))
    print(f"Loaded {len(records)} records from {path.name}")
    print(f"Seeds: {sorted(set(r['seed'] for r in records))}")
    print(f"Policies: {sorted(set(r['policy'] for r in records))}")
    print()

    # Build (seed, traj, policy, budget) lookup
    bt = defaultdict(dict)  # (seed, traj) -> (policy, budget) -> record
    for r in records:
        bt[(r["seed"], r["traj_id"])][(r["policy"], round(r["budget"], 2))] = r

    n_pairs = len(bt)
    print(f"Unique (seed, traj) pairs: {n_pairs}")

    # --- Per-(policy, budget) summary ---
    groups = defaultdict(list)
    for r in records:
        groups[(r["policy"], round(r["budget"], 2))].append(r)

    print()
    print("=" * 115)
    print("PER-(POLICY, BUDGET) SUMMARY (TRUE total cost including prep)")
    print("=" * 115)
    print(f"{'Policy':<12} {'bud':>5} {'n':>4} {'relL2':>10} {'CI95':>8} {'frac':>6} "
          f"{'neural_ms':>9} {'solver_ms':>9} {'ctrl_ms':>8} {'prep_ms':>8} {'TRUE_total_ms':>13}")
    for (pol, bud), recs in sorted(groups.items()):
        rel = np.array([r["rel_l2_mean"] for r in recs])
        frac = np.array([r["intervention_fraction"] for r in recs])
        if pol in ("G_TWOPASS", "G_ONLINE"):
            true_total = np.array([r["true_total_ms"] for r in recs])
            prep = np.array([r["prep_ms"] for r in recs])
            if pol == "G_TWOPASS":
                neu = np.array([r.get("exec_total_ms", 0) for r in recs])
                sol = np.array([0 for _ in recs])  # not separately stored for twopass
            else:
                neu = np.array([r.get("exec_neural_ms", 0) for r in recs])
                sol = np.array([r.get("exec_solver_ms", 0) for r in recs])
            ctrl = np.array([0 for _ in recs])  # not separately for brace
        else:
            true_total = np.array([r["total_wall_s"] * 1000 for r in recs])
            prep = np.array([0 for _ in recs])
            neu = np.array([r["neural_wall_s"] * 1000 for r in recs])
            sol = np.array([r["solver_wall_s"] * 1000 for r in recs])
            ctrl = np.array([r["controller_wall_s"] * 1000 for r in recs])
        print(f"{pol:<12} {bud:>5.2f} {len(recs):>4} {rel.mean():>10.5f} {ci95(rel):>8.5f} "
              f"{frac.mean():>6.3f} {neu.mean():>9.2f} {sol.mean():>9.2f} "
              f"{ctrl.mean():>8.2f} {prep.mean():>8.2f} {true_total.mean():>13.2f}")

    # --- Paired comparisons at 3 levels ---
    print()
    print("=" * 115)
    print("PAIRED COMPARISONS: diff = first - second (negative = second more accurate)")
    print("=" * 115)
    comps = [
        ("G_ONLINE", 0.2, "A", 0.0, "G_ONLINE(0.2) - A"),
        ("G_ONLINE", 0.4, "A", 0.0, "G_ONLINE(0.4) - A"),
        ("G_ONLINE", 0.6, "A", 0.0, "G_ONLINE(0.6) - A"),
        ("G_TWOPASS", 0.2, "A", 0.0, "G_TWOPASS(0.2) - A"),
        ("C", 0.05, "G_ONLINE", 0.2, "C(nom0.05) - G_ONLINE(0.2)"),
        ("C", 0.05, "G_TWOPASS", 0.2, "C(nom0.05) - G_TWOPASS(0.2)"),
        ("G_ONLINE", 0.2, "G_TWOPASS", 0.2, "G_ONLINE - G_TWOPASS at 0.2"),
        ("F", 0.05, "C", 0.05, "F vs C at nom0.05"),
    ]

    for polA, budA, polB, budB, label in comps:
        ka = (polA, round(budA, 2)); kb = (polB, round(budB, 2))
        diffs = []
        seed_diffs = defaultdict(list)
        traj_diffs = defaultdict(list)
        for (s, tid), d in bt.items():
            if ka in d and kb in d:
                diff = d[ka]["rel_l2_mean"] - d[kb]["rel_l2_mean"]
                diffs.append(diff)
                seed_diffs[s].append(diff)
                traj_diffs[tid].append(diff)

        n, md, sd, lo, hi, t, p, cohen = paired(np.array(diffs))
        # Seed level
        seed_means = np.array([np.mean(v) for v in seed_diffs.values()])
        ns, mds, _, los, his, _, ps, _ = paired(seed_means)
        # Traj level
        traj_means = np.array([np.mean(v) for v in traj_diffs.values()])
        nt, mdt, _, lot, hit, _, pt, _ = paired(traj_means)

        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        print(f"{label:<34} pair: n={n} diff={md:+.5f} [{lo:+.5f},{hi:+.5f}] |d|={abs(cohen):.3f} p={p:.2e} {sig}")
        print(f"{'':<34} seed: n={ns} diff={mds:+.5f} [{los:+.5f},{his:+.5f}] p={ps:.2e} | traj: n={nt} diff={mdt:+.5f} [{lot:+.5f},{hit:+.5f}] p={pt:.2e}")

    # --- Cost comparison at matched intervention fraction ---
    print()
    print("=" * 115)
    print("COST COMPARISON AT MATCHED ACTUAL INTERVENTION FRACTION (~0.20)")
    print("=" * 115)
    for pol, bud, label in [("C", 0.05, "C(nom 0.05, act~0.20)"),
                             ("G_TWOPASS", 0.2, "G_TWOPASS(0.2)"),
                             ("G_ONLINE", 0.2, "G_ONLINE(0.2)")]:
        recs = [r for r in records if r["policy"] == pol and abs(r["budget"] - bud) < 1e-9]
        rel = np.mean([r["rel_l2_mean"] for r in recs])
        if pol in ("G_TWOPASS", "G_ONLINE"):
            cost = np.mean([r["true_total_ms"] for r in recs])
        else:
            cost = np.mean([r["total_wall_s"] * 1000 for r in recs])
        frac = np.mean([r["intervention_fraction"] for r in recs])
        print(f"  {label:<28} relL2={rel:.5f} frac={frac:.3f} true_total_ms={cost:.2f}")


if __name__ == "__main__":
    main()
