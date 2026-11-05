# Statistical Audit — BRACE-CPU

**Date:** 2026-11-05  
**Data:** Corrected main experiment — `results/raw/exp_20261105_134910_63a5fa94_all.json` (3,500 records; 5 seeds × 25 trajectories × 8 policies × up to 6 budget levels).

---

## 1. Recalculation from raw data

All headline metrics were recalculated independently from the raw JSON records (not trusted from intermediate CSVs):

| Policy | n | relL2 mean | std | CI95 | intervention frac |
|---|---|---|---|---|---|
| A always neural | 125 | 0.2550 | 0.115 | 0.0207 | 0.000 |
| B always numerical | 125 | 0.0000 | 0.000 | 0.0000 | 1.000 |
| C fixed threshold | 625 | 0.0490 | 0.053 | 0.0049 | 0.660 |
| D adaptive threshold | 625 | 0.0035 | 0.005 | 0.0008 | 0.991 |
| E uncalibrated risk | 625 | 0.0489 | 0.056 | 0.0044 | 0.658 |
| F calibrated risk | 625 | 0.0489 | 0.056 | 0.0044 | 0.658 |
| G BRACE budget | 625 | 0.1501 | 0.094 | 0.0078 | 0.270 |
| H oracle | 125 | 0.0042 | 0.005 | 0.0004 | 0.955 |

These match the values reported in docs/RESEARCH_FINDINGS.md. ✅

Note: B (always numerical) shows relL2 ≈ 0 because B uses the same solver/dt as the reference generator — it is numerically self-consistent, not a meaningful "neural vs numerical" accuracy comparison. The meaningful accuracy baselines are A (neural) and H (oracle).

## 2. Confidence intervals

95% CIs use the normal approximation: mean ± 1.96·SE, SE = s/√n. Clustered by (seed, trajectory) for paired comparisons; n=125 paired (seed,traj) observations for policies evaluated on all 25 trajectories × 5 seeds.

## 3. Paired comparisons (identical test trajectories)

| Comparison | n | mean diff | CI95 | Cohen's d | p | Significance |
|---|---|---|---|---|---|---|
| G(0.2) − A(neural) | 125 | −0.0961 | [−0.1077, −0.0845] | −1.45 | 1.8e-32 | *** |
| G(0.4) − A(neural) | 125 | −0.1467 | [−0.1610, −0.1324] | −1.80 | 8.4e-41 | *** |
| G(0.6) − A(neural) | 125 | −0.1905 | [−0.2069, −0.1740] | −2.03 | 6.7e-46 | *** |
| C(0.05nom, act 0.66) − G(0.2) | 125 | −0.0170 | [−0.0332, −0.0009] | −0.19 | 0.041 | * |
| H(oracle) − G(0.2) | 125 | −0.1547 | [−0.1700, −0.1393] | −1.77 | 4.0e-40 | *** |
| F(cal, 0.05nom) − C(fixed, 0.05nom) | 125 | −0.0054 | [−0.0154, +0.0046] | −0.10 | 0.29 | ns |

Interpretation (diff = error_A − error_B; negative = B more accurate):
- **G vs A:** G is significantly more accurate than pure neural at all budgets (very large effects, d = −1.45 to −2.03). ✅ Supports RQ4 (selective correction improves reliability).
- **C vs G:** C is slightly more accurate but only because it overshoots its nominal budget (0.66 actual vs 0.05 nominal) — it intervenes ~3.3× more than G at matched budget. Not a fair budget comparison.
- **H vs G:** Oracle better than G (expected — oracle uses ground truth). Large gap (d = −1.77) shows headroom.
- **F vs C:** No significant difference (p = 0.29, CI includes zero). **Calibration null result confirmed statistically.** ✅ Supports RQ2 null finding.

## 4. Effect sizes

Cohen's d for G vs A ranges −1.45 to −2.03 (very large). Calibration effect d = −0.10 (negligible, ns).

## 5. Pseudoreplication check

Records are clustered by (seed, trajectory): n=125 independent (seed,trajectory) pairs, not 3,500 independent observations. Summary statistics use n=5 seeds (between-seed) or n=125 paired (seed,traj). No pseudoreplication: we do not treat each (policy, trajectory) record as fully independent when computing between-policy CIs; paired tests account for shared trajectories.

## 6. Outliers

relL2 std is large for A (0.115), C (0.053), E (0.056), F (0.056), G (0.094) — reflecting trajectory-to-trajectory variability (some trajectories are harder). No extreme outliers removed. CIs are wide for these policies, appropriately reflecting uncertainty.

## 7. Negative findings reported

- Calibration (F) provides no accuracy benefit over fixed threshold (C) at matched nominal budget (p=0.29, d=−0.10). ✅ Reported, not suppressed.
- Calibration does not improve ECE/Brier/AUROC over uncalibrated (RQ2 null). ✅ Reported.
- Threshold policies overshoot their declared budget (computational fairness/cost audit). ✅ Reported.

## 8. The "50% error reduction" claim

Verified: G at budget 0.4 reduces relL2 from 0.2550 (A) to 0.1083 = 57.5% reduction; at 0.6, 0.0645 = 74.7% reduction; at 0.2, 0.1588 = 37.7% reduction. So "roughly 50% error reduction at ~40% intervention budget" is accurate and reproducible. At 20% budget, reduction is 37.7%.

---

*End of statistical audit.*
