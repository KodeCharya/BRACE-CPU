# Publication Statistical Review — BRACE-CPU

**Date:** 2026-11-05  
**Data:** Corrected main experiment `results/raw/exp_20261105_134910_63a5fa94_all.json`  
**Script:** `scripts/stat_review.py`

---

## 1. Experimental units (independence assessment)

| Unit | Count | Independent? | Notes |
|---|---|---|---|
| Seeds | 5 (0–4) | Yes | Independent model initializations + data shuffling |
| Trajectories per seed | 25 | Conditionally | Share the same model weights within a seed; correlated through the model |
| (seed, trajectory) pairs | 125 | Mostly | Best available clustering unit for paired comparisons |
| Timesteps per trajectory | 60 | No | Autocorrelated; never treated as independent in this analysis |
| Total records | 3,500 | No | 5 seeds × 25 traj × 8 policies × up to 6 budget levels |

**Key point:** 3,500 records ≠ 3,500 independent samples. The analysis clusters by (seed, trajectory), giving n = 125 paired observations for policy comparisons. Timesteps within a trajectory are correlated and were never treated as independent (all metrics are per-trajectory means).

The truly independent replicates are the 5 seeds. With only 5 seeds, between-seed uncertainty is estimated with 4 degrees of freedom; within-seed trajectory pairing increases power for paired comparisons but does not create new independent models.

---

## 2. Recalculated summary statistics (corrected raw data)

| Policy | n (pairs) | rel L2 mean | SD | CI95 | Intervention frac | Total ms |
|---|---|---|---|---|---|---|
| A always neural | 125 | 0.2550 | 0.1179 | 0.0207 | 0.000 | 8.50 |
| B always numerical | 125 | 0.0000 | 0.0000 | 0.0000 | 1.000 | 10.98 |
| C fixed threshold | 625 | 0.0490 | 0.0623 | 0.0049 | 0.660 | 20.87 |
| D adaptive threshold | 625 | 0.0035 | 0.0095 | 0.0008 | 0.991 | 26.69 |
| E uncalibrated risk | 625 | 0.0489 | 0.0652 | 0.0051 | 0.658 | 56.07 |
| F calibrated risk | 625 | 0.0489 | 0.0652 | 0.0051 | 0.658 | 66.89 |
| G BRACE budget | 625 | 0.1501 | 0.1000 | 0.0078 | 0.270 | 9.27 |
| H oracle | 125 | 0.0042 | 0.0024 | 0.0004 | 0.955 | 21.70 |

Note: n = 625 for budget-varied policies (5 budgets × 125 pairs); n = 125 for single-setting policies. C/E/F aggregate across budgets; see evidence audit D1 for the mislabeling issue.

G per-budget (n = 125 each):

| Budget | rel L2 mean | SD | CI95 | frac | Total ms | Error reduction vs A |
|---|---|---|---|---|---|---|
| 0.05 | 0.2217 | 0.0996 | 0.0175 | 0.050 | 8.67 | 13.0% |
| 0.10 | 0.1970 | 0.0926 | 0.0162 | 0.100 | 8.82 | 22.7% |
| 0.20 | 0.1588 | 0.0870 | 0.0152 | 0.200 | 9.09 | 37.7% |
| 0.40 | 0.1083 | 0.0712 | 0.0128 | 0.400 | 9.63 | 57.5% |
| 0.60 | 0.0645 | 0.0480 | 0.0088 | 0.600 | 10.15 | 74.7% |

---

## 3. Paired comparisons (identical test trajectories, n = 125 seed-trajectory pairs)

Sign convention: diff = error(first) − error(second). Negative = second policy more accurate (lower error). Negative Cohen's d = second policy favored.

| Comparison | n | mean diff | CI95 | Cohen's d | t-stat | p-value | Interpretation |
|---|---|---|---|---|---|---|---|
| G(0.2) − A(neural) | 125 | −0.0961 | [−0.1077, −0.0845] | −1.45 | −16.2 | p < 0.001 | G much better than neural |
| G(0.4) − A(neural) | 125 | −0.1467 | [−0.1610, −0.1324] | −1.80 | −20.1 | p < 0.001 | G much better than neural |
| G(0.6) − A(neural) | 125 | −0.1905 | [−0.2069, −0.1740] | −2.03 | −22.6 | p < 0.001 | G much better than neural |
| C(nom 0.05, act 0.20) − G(0.2) | 125 | −0.0170 | [−0.0332, −0.0009] | −0.19 | −2.07 | p = 0.041 | C slightly better, but intervenes 3× its nominal budget |
| H(oracle) − G(0.2) | 125 | −0.1547 | [−0.1700, −0.1393] | −1.77 | −20.0 | p < 0.001 | Oracle better (uses ground truth) |
| F(cal) − C(fixed) at nom 0.05 | 125 | −0.0054 | [−0.0154, +0.0046] | −0.10 | −1.06 | p = 0.29 | **No difference** (calibration null) |
| E(uncal) − F(cal) at nom 0.05 | 125 | 0.0000 | [0, 0] | 0.00 | 0.0 | p = 1.00 | E and F identical |

**Cohen's d interpretation:** |d| > 0.8 = large effect. G vs A shows very large effects (|d| = 1.45–2.03). Calibration effect is negligible (|d| = 0.10, ns).

---

## 4. Statistical methodology assessment

### 4.1 Tests used
- Paired t-test on (seed, trajectory) differences. Appropriate because the same trajectories are evaluated under all policies within a seed, controlling for trajectory difficulty.
- 95% CIs: normal approximation (mean ± 1.96·SE), SE = SD/√n.

### 4.2 Multiple comparisons
Six pairwise comparisons are reported. Without correction, the family-wise error rate is inflated. The strongest results (G vs A, p < 1e-30) remain significant under any correction (Bonferroni: α = 0.05/6 ≈ 0.008). The marginal result (C vs G, p = 0.041) would NOT survive Bonferroni correction (0.041 > 0.008) and should be described as "nominally significant, not correction-robust."

### 4.3 p-values below 1e-30
The reported exact values (1.8e-32, 8.4e-41, 6.7e-46) are at double-precision limits. t-statistics are −16.2, −20.1, −22.6 with df = 124. These are far below any conventional threshold. **Recommendation:** Report as "p < 0.001" with t-statistics, rather than exact sub-1e-30 values, to avoid overstating numerical precision.

### 4.4 Clustering limitation
Trajectories within a seed share the same model weights, so they are not fully independent. n = 125 paired observations is optimistic relative to n = 5 independent seeds. The paired design is reasonable for within-seed comparisons, but between-seed generalization rests on only 5 seeds. CIs should be interpreted with this limitation in mind (documented in limitations).

### 4.5 Outliers
No outliers removed. rel L2 SD is large for neural (0.118), C (0.062), E/F (0.065), G (0.100), reflecting trajectory-to-trajectory variability. Wide CIs appropriately reflect this uncertainty.

---

## 5. Calibration analysis (RQ2)

| Metric | Calibrated | Uncalibrated | Difference |
|---|---|---|---|
| ECE | 0.0096 ± 0.0020 | 0.0103 ± 0.0011 | −0.0007 (ns) |
| Brier score | 0.0251 ± 0.0028 | 0.0251 ± 0.0027 | 0.0000 |
| AUROC | 0.9879 ± 0.0021 | 0.9879 ± 0.0021 | 0.0000 |

- **Discrimination (AUROC):** Excellent (0.988), identical calibrated vs uncalibrated (calibration is monotonic in the score, so ranking is preserved).
- **Calibration (ECE/Brier):** Essentially identical. No measurable calibration benefit.
- **Paired F vs C at nominal 0.05:** p = 0.29, d = −0.10, CI includes zero. Calibration provides no accuracy benefit over fixed threshold at matched nominal budget.
- **Calibration-set size ablation:** ECE/Brier stable across 25%/50%/100% calibration-set sizes; larger sets do not improve calibration.

**Conclusion:** High AUROC does NOT imply good calibration, but here both discrimination and calibration are already strong in the uncalibrated score; Platt scaling adds nothing. The null result is robust and should be reported prominently.

---

## 6. Budget analysis (RQ3)

| Policy | Nominal budget | Actual intervention frac | Total ms/rollout | Notes |
|---|---|---|---|---|
| G BRACE | 0.05 | 0.050 | 8.67 | Enforces exactly |
| G BRACE | 0.10 | 0.100 | 8.82 | Enforces exactly |
| G BRACE | 0.20 | 0.200 | 9.09 | Enforces exactly |
| G BRACE | 0.40 | 0.400 | 9.63 | Enforces exactly |
| G BRACE | 0.60 | 0.600 | 10.15 | Enforces exactly |
| C fixed threshold | 0.05 (nominal) | 0.202 (actual) | 15.62 | Overshoots 4× |
| C fixed threshold | 0.10 (nominal) | 0.452 (actual) | 18.44 | Overshoots 4.5× |
| C fixed threshold | 0.20 (nominal) | 0.704 (actual) | 21.39 | Overshoots 3.5× |
| E uncalibrated | 0.05 (nominal) | 0.224 (actual) | 50.85 | Overshoots 4.5×; high controller cost |
| F calibrated | 0.05 (nominal) | 0.224 (actual) | 61.61 | Overshoots 4.5×; highest cost |

**Key finding:** G (BRACE) enforces the intervention budget exactly. Threshold policies (C, E, F) overshoot their nominal budgets under error accumulation. Comparing at matched *actual* intervention fraction (~0.20): C rel L2 = 0.1418 vs G rel L2 = 0.1588 — C is slightly more accurate but only because it violates its nominal budget (intervenes 0.20 vs nominal 0.05).

**Cost fairness:** E and F incur high per-step controller cost (48–59 ms/rollout) from risk computation, with no accuracy benefit over C (21 ms) — they are dominated. G has low online controller cost (0.13 ms) plus a one-time preparation cost (~54 ms, reported separately).

---

## 7. Conclusion on statistical validity

The statistical analysis is fundamentally sound: paired design, appropriate clustering by (seed, trajectory), CI95s reported, null results (calibration) honestly reported, effect sizes provided. The main corrections needed are:
1. Report p < 0.001 instead of exact sub-1e-30 values.
2. Clarify Cohen's d sign convention (or report |d| with direction in words).
3. Fix the C/E/F table mislabeling (aggregate vs nominal budget) — see evidence audit D1.
4. Add multiple-comparison correction note (Bonferroni) for the marginal C vs G result.
5. Acknowledge the n = 5 independent seeds limitation for between-seed generalization.

*End of statistical review.*
