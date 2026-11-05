# Publication Evidence Audit — BRACE-CPU

**Date:** 2026-11-05  
**Authoritative experiment:** `results/raw/exp_20261105_134910_63a5fa94_all.json` (corrected rerun, 3,500 records)  
**Superseded experiment:** `results/raw/exp_20261105_132231_985e2bf7_all.json` (defective code; NOT used)  
**Reproducibility experiment:** `results/raw/exp_20261105_135732_33b0c6f7_all.json` (seed-0 retrain, 700 records)  
**Verification script:** `scripts/evidence_audit.py`, `scripts/stat_review.py`

---

## 1. Experiment inventory

| Experiment | Records | Status | Used in manuscript? |
|---|---|---|---|
| exp_20261105_134910_63a5fa94 | 3,500 | **Authoritative (corrected)** | YES |
| exp_20261105_132231_985e2bf7 | 3,500 | Superseded (defective `times[]` + uncosted controller) | NO |
| exp_20261105_135732_33b0c6f7 | 700 | Reproducibility (seed 0 retrain) | NO (supports reproducibility claim) |
| exp_20261105_131700_541c06cf | 1,260 | Pilot (3 seeds) | NO |

All raw records are immutable; none were modified during this audit.

---

## 2. Evidence traceability table

| # | Manuscript claim | Source file | Metric | Calculation | Raw evidence | Status |
|---|---|---|---|---|---|---|
| 1 | A neural rel L2 = 0.2550 ± 0.0207 | `paper/main.tex` tab:summary | rel L2 mean, CI95 | mean ± 1.96·SE over 125 (seed,traj) pairs | `exp_20261105_134910` A_always_neural | **Supported** (recalculated 0.25495 ± 0.02067) |
| 2 | B numerical rel L2 ≈ 0 | `paper/main.tex` tab:summary | rel L2 mean | mean over 125 pairs | `exp_20261105_134910` B_always_numerical | **Supported** (0.00000; degenerate—same solver as reference) |
| 3 | H oracle rel L2 = 0.0042 ± 0.0004 | `paper/main.tex` tab:summary | rel L2 mean, CI95 | mean ± 1.96·SE over 125 pairs | `exp_20261105_134910` H_oracle | **Supported** (0.00416 ± 0.00042) |
| 4 | G(0.05) rel L2 = 0.2217 ± 0.0175 | `paper/main.tex` tab:summary | rel L2 mean, CI95 | mean ± 1.96·SE over 125 pairs | `exp_20261105_134910` G_brace_budget@0.05 | **Supported** (0.22171 ± 0.01748) |
| 5 | G(0.10) rel L2 = 0.1970 ± 0.0162 | `paper/main.tex` tab:summary | rel L2 mean, CI95 | mean ± 1.96·SE over 125 pairs | `exp_20261105_134910` G_brace_budget@0.10 | **Supported** (0.19696 ± 0.01618) |
| 6 | G(0.20) rel L2 = 0.1588 ± 0.0152 | `paper/main.tex` tab:summary | rel L2 mean, CI95 | mean ± 1.96·SE over 125 pairs | `exp_20261105_134910` G_brace_budget@0.20 | **Supported** (0.15884 ± 0.01515) |
| 7 | G(0.40) rel L2 = 0.1083 ± 0.0128 | `paper/main.tex` tab:summary | rel L2 mean, CI95 | mean ± 1.96·SE over 125 pairs | `exp_20261105_134910` G_brace_budget@0.40 | **Supported** (0.10827 ± 0.01271) |
| 8 | G(0.60) rel L2 = 0.0645 ± 0.0088 | `paper/main.tex` tab:summary | rel L2 mean, CI95 | mean ± 1.96·SE over 125 pairs | `exp_20261105_134910` G_brace_budget@0.60 | **Supported** (0.06450 ± 0.00880) |
| 9 | C fixed threshold rel L2 = 0.0490 ± 0.0049, frac = 0.660 | `paper/main.tex` tab:summary | rel L2, frac, CI | **AGGREGATE across all 5 budgets**, NOT at nominal 0.05 | `exp_20261105_134910` C_fixed_threshold (all budgets) | **MISLABELED** — table says "nominal 0.05" but shows aggregate (actual frac 0.66 ≠ 0.05). At actual budget 0.05: rel L2 = 0.1418 ± 0.0092, frac = 0.202. |
| 10 | E uncalibrated rel L2 = 0.0489 ± 0.0051, frac = 0.658 | `paper/main.tex` tab:summary | rel L2, frac, CI | **AGGREGATE across all 5 budgets** | `exp_20261105_134910` E_uncalibrated_risk (all budgets) | **MISLABELED** — same issue as #9. At actual budget 0.05: rel L2 = 0.1364 ± 0.0110, frac = 0.224. |
| 11 | F calibrated rel L2 = 0.0489 ± 0.0051, frac = 0.658 | `paper/main.tex` tab:summary | rel L2, frac, CI | **AGGREGATE across all 5 budgets** | `exp_20261105_134910` F_calibrated_risk (all budgets) | **MISLABELED** — same issue as #9. At actual budget 0.05: rel L2 = 0.1364 ± 0.0110, frac = 0.224. |
| 12 | G vs A: d = −1.45 to −2.03, p < 1e-30 | `paper/main.tex` sec:results-reliability | Cohen's d, p-value | Paired t-test on 125 (seed,traj) pairs | `exp_20261105_134910` | **Supported** (G(0.2): d=−1.45,p<1e-30; G(0.4): d=−1.80; G(0.6): d=−2.03) |
| 13 | H vs G: d = −1.77, p < 1e-30 | `paper/main.tex` (implied) | Cohen's d, p-value | Paired t-test on 125 pairs | `exp_20261105_134910` | **Supported** (d=−1.77,p<1e-30) |
| 14 | C vs G at matched ~20% frac: C better (0.049 vs 0.159) | `paper/main.tex` sec:results-budget | rel L2 | **Uses aggregate C value (0.049), not matched-frac value** | `exp_20261105_134910` | **PARTIALLY SUPPORTED** — C at actual frac ~0.20 has rel L2 = 0.1418 (not 0.049). C is slightly better than G (0.142 vs 0.159) but violates its nominal budget (intervenes 0.20 vs nominal 0.05). |
| 15 | Calibration null: ECE 0.0096 vs 0.0103, Brier 0.0251 both, AUROC identical | `paper/main.tex` tab:calibration | ECE, Brier, AUROC | mean ± std over 5 seeds | `exp_20261105_134910` (risk/calibration eval) + ablation | **Supported** (recalibrated: ECE 0.0096±0.0020 vs 0.0103±0.0011; Brier identical; AUROC identical 0.9879) |
| 15b | F vs C paired p = 0.29, d = −0.10, CI [−0.0154,+0.0046] | `paper/main.tex` sec:results-calibration | p-value, Cohen's d, CI | Paired t-test, 125 pairs | `exp_20261105_134910` F vs C at nominal 0.05 | **Supported** (recalibrated p=0.292, d=−0.095, CI [−0.0154,+0.0046]) |
| 16 | E and F identical results | `paper/main.tex` (implied by identical table rows) | rel L2 | Direct comparison | `exp_20261105_134910` | **Supported** (E and F bit-identical at nominal 0.05) |
| 17 | ~50% error reduction at 40% budget | `paper/main.tex` sec:results-acc-cost | % reduction | (0.2550 − 0.1083)/0.2550 = 57.5% | `exp_20261105_134910` A vs G(0.40) | **Supported** (57.5% reduction; "roughly 50%" is accurate) |
| 18 | G enforces budget exactly (0.05→0.050 … 0.60→0.600) | `paper/main.tex` sec:results-budget | intervention fraction | mean over 125 pairs per budget | `exp_20261105_134910` G_brace_budget | **Supported** (all budgets: actual frac = nominal) |
| 19 | C/E/F overshoot nominal 0.05 → actual ~0.66 | `paper/main.tex` sec:results-budget | intervention fraction | mean over 625 records (all budgets) | `exp_20261105_134910` | **PARTIALLY SUPPORTED** — 0.66 is the cross-budget aggregate, not the actual-0.05-subset value (0.20–0.22). The claim "targeting 5% yields ~66%" is misleading: targeting 5% actually yields ~20%; 0.66 results from averaging across budgets 0.05–0.60. |
| 20 | AUROC 0.9879 ± 0.0021 | `paper/main.tex` sec:results-calibration | AUROC | mean ± std over 5 seeds | Risk/calibration evaluation (5 seeds) | **Supported** |
| 21 | OOD fail rate ~0.67 vs ~0.09 in-dist; AUROC ~0.90 | `paper/main.tex` sec:results-robustness | fail rate, AUROC | OOD evaluation (seeds 0–2) | `scripts/run_ablation.py` output | **Supported** |
| 22 | Ablation: residual AUROC 0.979, mass 0.58, etc. | `paper/main.tex` sec:ablation | AUROC | Feature ablation (seeds 0–2) | `scripts/run_ablation.py` output | **Supported** |
| 23 | ~17K parameters | `paper/main.tex` sec:method | parameter count | MLP architecture | `src/models/mlp.py` | **Supported** (16,768) |
| 24 | B (always numerical) is degenerate (same solver as reference) | `paper/main.tex` tab:summary note | — | Code inspection | `src/solvers/burgers.py`, generator | **Supported** |

---

## 3. Discrepancies requiring manuscript correction

### Discrepancy D1 (critical): C/E/F table rows mislabeled
The manuscript table labels C, E, F rows as budget "0.05 (nom.)" but reports values (rel L2 ≈ 0.049, frac ≈ 0.66) that are the **aggregate across all five budgets** (n = 625), not the performance at actual budget 0.05. At actual budget 0.05 (n = 125), C rel L2 = 0.1418, frac = 0.202.

**Fix:** Either (a) relabel rows as "all budgets (aggregate)" and report frac honestly, or (b) report C/E/F specifically at their nominal-0.05 runs (rel L2 ≈ 0.14, frac ≈ 0.20) and separately note budget overshoot. Option (b) is fairer for the budget-enforcement comparison.

### Discrepancy D2: "C better than G at matched ~20%" uses wrong C value
Manuscript text (sec:results-budget) says "At a matched actual intervention fraction (~20%), threshold policies are slightly more accurate than G (rel L2 ≈ 0.049 vs 0.159)." The 0.049 figure is the cross-budget aggregate, not the matched-fraction value. At actual matched frac ~0.20, C rel L2 = 0.1418 (not 0.049). C is still slightly better than G (0.142 vs 0.159) but only by violating its nominal 0.05 budget.

**Fix:** Correct to rel L2 ≈ 0.142 vs 0.159 and clarify that C achieves this by intervening ~3× its nominal budget.

### Discrepancy D3: p-values reported as exact below 1e-30
The p-values (1.8e-32, 8.4e-41, 6.7e-46) are at/below double-precision underflow for the t-statistics involved. Reporting exact values to 2 significant digits is not numerically meaningful.

**Fix:** Report as "p < 0.001" or give t-statistics, noting the values are far below conventional thresholds.

### Discrepancy D4: Negative Cohen's d sign convention
Manuscript reports "d = −1.45 to −2.03" for G vs A. The negative sign arises because diff = error_A − error_B (negative = B more accurate). This is confusing for readers expecting a positive effect size for "G improves over A."

**Fix:** Clarify sign convention, or report |d| with direction stated in words (e.g., "large effect favoring G, |d| = 1.45–2.03").

### Discrepancy D5: Duplicate `\section{Conclusion}`
`paper/main.tex` has a complete Conclusion (lines 169–194) AND a placeholder "[To be completed...]" (line 210). The placeholder must be removed.

---

## 4. Claim classification summary

| Classification | Claims |
|---|---|
| **Supported** | #1–#8 (A/B/H/G table values), #12–#13 (G vs A, H vs G significance), #15–#16 (calibration null, E≡F), #17 (50% reduction), #18 (G budget exact), #20–#24 |
| **Partially supported** | #9–#11 (C/E/F table values are real but mislabeled as "nominal 0.05"), #14 (C vs G at matched frac—direction correct but C value wrong), #19 (budget overshoot—direction correct but 0.66 is aggregate not actual-0.05) |
| **Unsupported** | None (all quantitative claims trace to raw data) |
| **Inconclusive** | None |

---

## 5. Conclusion

All quantitative claims in the manuscript trace to the corrected raw records. The only defects are **presentation/labeling** (D1–D5), not data fabrication. No claim is unsupported. The calibration null result, E≡F identity, and budget-overshoot finding are all confirmed and should remain prominently reported.

*End of evidence audit.*
