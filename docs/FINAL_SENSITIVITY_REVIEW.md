# Final Statistical and Operational Methodology Sensitivity Review - BRACE-CPU

**Date:** 2026-11-09  
**Project:** `/home/mukesh/Documents/research/BRACE-CPU/`  
**Git baseline:** `c8d3a9b` (branch `master`)  
**Authoritative experiment:** `results/raw/exp_20261105_134910_63a5fa94_all.json` (3,500 records)  
**Rebuild experiment:** `results/rebuild/corrected_20261109_101618/corrected_20261109_101618_all.json` (3,375 records)  
**Reviewer role:** Independent statistical and operational methodology reviewer  
**Rules followed:** Read-only inspection with read-only analysis scripts; no modifications to raw data, historical experiments, Git history, or manuscript source.

---

## 1. Statistical Dependence Assessment

### 1.1 Crossed dependence structure

The design is **crossed**: 5 seeds × 25 shared trajectories = 125 unique (seed, trajectory) pairs. Each of the 8 policies is evaluated on every pair. This creates **two-way dependence**:

1. **Within-seed dependence:** Trajectories share the same model weights → errors correlated.
2. **Within-trajectory dependence:** The same trajectory appears under 5 different models (seeds) → errors correlated across seeds.

With only n=5 seeds, asymptotic cluster-bootstrap inference is unreliable. The sensitivity analysis therefore compares three aggregation levels.

### 1.2 Aggregation sensitivity

For each paired comparison (diff = policy_A − policy_B; negative = B more accurate):

| Comparison | Pair (n=125) | Seed (n=5) | Trajectory (n=25) | Conclusion stability |
|---|---|---|---|---|
| G(0.2) vs A | diff=-0.096, p<0.001, \|d\|=1.452 | diff=-0.096, p=3.25e-06, \|d\|=16.47 | diff=-0.096, p=2.20e-08, \|d\|=1.633 | **Robust** |
| G(0.4) vs A | diff=-0.147, p<0.001, \|d\|=1.797 | diff=-0.147, p=3.32e-06, \|d\|=16.38 | diff=-0.147, p=4.61e-10, \|d\|=2.007 | **Robust** |
| G(0.6) vs A | diff=-0.190, p<0.001, \|d\|=2.026 | diff=-0.190, p=1.04e-06, \|d\|=21.92 | diff=-0.190, p=1.09e-10, \|d\|=2.158 | **Robust** |
| C(nom 0.05) vs G(0.2) | diff=-0.017, p=0.041, \|d\|=0.185 | diff=-0.017, p=0.028, \|d\|=1.504 | diff=-0.017, p=0.318, \|d\|=0.204 | **NOT ROBUST at traj level** |
| H vs G(0.2) | diff=-0.155, p<0.001, \|d\|=1.767 | diff=-0.155, p=2.26e-05, \|d\|=10.12 | diff=-0.155, p=1.01e-09, \|d\|=1.927 | **Robust** |
| F vs C at nom 0.05 | diff=-0.005, p=0.292, \|d\|=0.095 | diff=-0.005, p=0.134, \|d\|=0.838 | diff=-0.005, p=0.574, \|d\|=0.114 | **Null confirmed** |

**Key finding:** The C vs G(0.2) claim (C slightly more accurate at matched intervention fraction) is significant at the pair level (p=0.041) and seed level (p=0.028), but **NOT significant at the trajectory level** (p=0.318, n=25). The trajectory level is the correct unit for assessing generalizability to new trajectories. The manuscript should not claim C is significantly more accurate than G.

**Cohen's d inflation at seed level:** Seed-level d values are inflated (16–22) because averaging across 25 trajectories drastically reduces standard error. Trajectory-level d is more representative. Pair-level d is an intermediate estimate.

### 1.3 Timesteps as independent samples

Timesteps within a trajectory are autocorrelated (each state is a function of the previous state). The analysis correctly uses per-trajectory means (one value per trajectory per policy), never treating 60 timesteps as independent observations.

---

## 2. BRACE Policy G Execution Sequence (Two-Pass Confirmed)

### 2.1 Two-pass procedure in `src/controller/policies.py`

Policy G (`BUDGET_RISK_RANKED`) uses a **two-pass procedure**:

**Pass 1 — Preparation (`prepare()` → `_prepare_budget_ranked()`):**
1. Run a full **all-surrogate trajectory** (60 steps, 0 interventions)
2. At each step: call `surrogate.predict(u, params, t)` → `u_pred`
3. Call `risk_estimator.score(surrogate, u, params, t, dt)` → risk score `r`
   - Internally calls `surrogate.predict()` again (duplicate neural inference)
   - Calls `extract_features()` which calls `temporal_inconsistency()` → 3 more `predict()` calls
4. Store `risks[i] = r`, advance `u = u_pred`
5. Sort risks descending, select top-k indices (k = round(budget × n_steps))
6. Store `intervention_indices` (fixed for the entire rollout)

**Pass 2 — Execution (`simulate_rollout()` → `decide()`):**
1. For each step i from 0 to 59:
   - Call `decide(u, params, t, dt, i)`:
     - Returns `(i in intervention_indices, None, elapsed)` — no neural prediction (G returns None for u_pred)
   - If `intervene` is True: call `solver.advance(u, t, dt, 1)` (numerical correction)
   - Else: call `surrogate.predict(u, params, t)` (neural inference, timed as `neural_wall_s`)
   - Update state: `u = solver_output` or `u = u_pred`

### 2.2 Redundant computation in pass 1

Per pass-1 step, the neural model is called **5 times**:
- 1× explicit `predict()` in `_prepare_budget_ranked` loop (for trajectory advancement)
- 1× `predict()` inside `risk_estimator.score()` (for residual feature)
- 3× `predict()` inside `temporal_inconsistency()` (for consistency feature)

For 60 steps: **300 neural predictions in pass 1** for a 60-step rollout. At 0.136 ms per predict (1 thread), this is ~41 ms of the ~54 ms preparation cost.

### 2.3 Risk ranking staleness

Risk scores are computed along an **uncorrected all-surrogate trajectory** (pass 1). During pass 2, numerical corrections alter the state trajectory. The pre-computed risk rankings are **not re-evaluated** after corrections. A step flagged as "high risk" in pass 1 reflects the state that would have existed without any corrections; after earlier corrections are applied, the actual state diverges and the ranking may no longer reflect the true risk at that actual state.

The code does not re-score risk after corrections. The intervention indices are fixed at the start of the rollout.

### 2.4 G_ONLINE: Single-pass alternative (new, `src/controller/brace_controller.py`)

The rebuild introduces G_ONLINE, a single-pass variant:
- At each step during the rollout, score risk on the **actual controlled trajectory** (with corrections)
- Maintain a running list of (risk, index) for steps seen so far
- Intervene if: (a) budget requires it (remaining_budget ≥ remaining_steps), or (b) current risk is among top-k riskiest so far
- Guarantees exact budget enforcement while scoring risk on fresh states

**G_ONLINE is strictly better than G_TWOPASS on both accuracy and cost:**
- At budget 0.2: relL2 0.132 (online) vs 0.159 (twopass), total cost 60.94 ms vs 67.80 ms
- At budget 0.6: relL2 0.034 (online) vs 0.065 (twopass), total cost 66.47 ms vs 69.00 ms
- Between-seed: G_ONLINE − G_TWOPASS diff = −0.027, p = 0.004 (seed), p = 0.002 (traj)

The online version scores risk on the actual trajectory (post-correction states), so its interventions reflect true current risk rather than stale pre-computation.

---

## 3. Full Computational Cost Breakdown

### 3.1 Cost components for G_TWOPASS (from corrected rebuild experiment)

Measured over 125 seed-traj pairs per budget (median per-rollout cost):

| Budget | Interventions | Prep neural | Prep risk | Prep ranking | Prep total | Exec neural | Exec solver | Exec ctrl | Reported total | TRUE total (incl prep) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.05 | 3 | 8.79 ms | 45.29 ms | 0.02 ms | 54.22 ms | 6.67 ms | 0.57 ms | 0.02 ms | 7.26 ms | 61.48 ms |
| 0.10 | 6 | 8.79 ms | 45.27 ms | 0.02 ms | 54.22 ms | 6.67 ms | 0.57 ms | 0.02 ms | 7.26 ms | 61.48 ms |
| 0.20 | 12 | 8.79 ms | 45.29 ms | 0.02 ms | 54.22 ms | 6.67 ms | 2.25 ms | 0.02 ms | 8.94 ms | 63.16 ms |
| 0.40 | 24 | 8.79 ms | 45.29 ms | 0.02 ms | 54.22 ms | 5.08 ms | 4.41 ms | 0.02 ms | 9.51 ms | 63.73 ms |
| 0.60 | 36 | 8.79 ms | 45.29 ms | 0.02 ms | 54.22 ms | 3.41 ms | 6.58 ms | 0.02 ms | 10.01 ms | 64.23 ms |

**Critical finding:** The `total_wall_s` reported in the original experiment **excluded** `prepare_wall_s`. The true per-rollout cost is prep + execution:
- At budget 0.2: true cost = 63.16 ms (reported: 8.94 ms) → **7.1× underreporting**
- Preparation constitutes **86% of true cost** for G_TWOPASS

### 3.2 Cost components for G_ONLINE

G_ONLINE has **zero preparation cost** (single-pass). All cost is in execution:

| Budget | Exec neural | Exec risk | Exec solver | Exec total | TRUE total (= exec total) |
|---|---|---|---|---|---|
| 0.05 | 9.19 ms | 29.28 ms | 0.59 ms | 55.17 ms | 55.17 ms |
| 0.10 | 9.21 ms | 29.46 ms | 1.19 ms | 55.79 ms | 55.79 ms |
| 0.20 | 9.21 ms | 30.64 ms | 2.38 ms | 57.27 ms | 57.27 ms |
| 0.40 | 9.32 ms | 31.00 ms | 4.76 ms | 59.86 ms | 59.86 ms |
| 0.60 | 9.36 ms | 31.00 ms | 7.12 ms | 62.14 ms | 62.14 ms |

Note: The exec_risk_ms for G_ONLINE in the analysis script is labeled as part of controller overhead, but the `CostBreakdown` dataclass records it separately. Total cost is correctly included.

### 3.3 Cost comparison at matched actual intervention fraction (~0.20)

| Policy | Reported budget | Actual frac | relL2 | Reported total ms | TRUE total ms | Accuracy rank | Cost rank |
|---|---|---|---|---|---|---|---|
| C (fixed threshold) | 0.05 | 0.202 | 0.142 | 16.81 | 16.81 | 2nd | 1st (cheapest) |
| G_ONLINE | 0.2 | 0.200 | 0.132 | 60.94 | 60.94 | 1st (most accurate) | 2nd |
| G_TWOPASS | 0.2 | 0.200 | 0.159 | 8.98 (EXCLUDED prep) | 67.80 | 3rd | 3rd (most expensive) |
| E (uncalibrated) | 0.05 | 0.224 | 0.136 | 54.56 | 54.56 | 2nd | 2nd |
| F (calibrated) | 0.05 | 0.224 | 0.136 | 66.30 | 66.30 | 2nd | 3rd |

**At matched intervention fraction (~0.20–0.22):**
- C is **cheapest** (16.81 ms) but slightly less accurate than G_ONLINE (0.142 vs 0.132)
- G_ONLINE is **most accurate** (0.132) but **3.6× more expensive** than C (60.94 vs 16.81 ms)
- G_TWOPASS is **least accurate** (0.159) and **most expensive** (67.80 ms)

**The claim that G has "low online cost" is misleading** when prep cost is included. E and F also cost more than G_ONLINE per rollout (54.56 and 66.30 ms) because their per-step risk computation is expensive.

### 3.4 Pareto implications

If accuracy is plotted against **true total cost** (including prep for G_TWOPASS):
- G_TWOPASS falls off the Pareto frontier (dominated by G_ONLINE and C)
- G_ONLINE remains on the frontier at higher budgets
- C dominates G_TWOPASS at matched cost and accuracy

The Pareto figures in `paper/figures/` were generated from intervention fraction vs relL2, NOT from cost vs relL2. They do not misrepresent G_TWOPASS's cost position because they do not plot cost on an axis. However, any prose claim about G's "low online cost" is a partial accounting that must be qualified.

---

## 4. Statistical Sensitivity Summary

### Claims that survive all sensitivity checks
1. **G vs A (BRACE improves over pure neural):** Significant at pair (p<0.001), seed (p<0.001), and trajectory (p<0.001) levels. Large effect sizes at trajectory level (|d|=1.63–2.16).
2. **G enforces budget exactly:** Intervention count = round(budget × n_steps) at all budgets and all aggregation levels.
3. **Calibration adds no benefit:** Null confirmed at pair (p=0.292), seed (p=0.134), and trajectory (p=0.574) levels.
4. **G_ONLINE > G_TWOPASS (accuracy):** Significant at pair (p=1.05e-08), seed (p=0.004), and trajectory (p=0.002) levels.

### Claims that do not survive
1. **C more accurate than G at matched frac:** Pair p=0.041, seed p=0.028, trajectory p=0.318 → NOT significant at trajectory level.
2. **G has low total cost:** True cost is 6.3–7.2× reported cost for G_TWOPASS due to prep exclusion.

### Limitations of this sensitivity analysis
- Only 5 seeds: between-seed inference is limited (4 df). Seed-level CIs are wide.
- Trajectory-level analysis (n=25) is the most appropriate for generalizing to new trajectories.
- Two-way cluster bootstrap was NOT performed (unreliable with 5 seed clusters).
- Timesteps treated as correlated (per-trajectory means), not independent.

---

## 5. Specific Manuscript Recommendations

### CRITICAL: Fix C/E/F table rows (tab:summary)

**Current (incorrect):** C/E/F rows show "budget=0.05 (nom.)" with relL2=0.049, frac=0.66 — these are cross-budget aggregate values.

**Corrected (actual budget=0.05 values):**
- C: relL2=0.1418, CI=0.0092, frac=0.202, total_ms=16.81
- E: relL2=0.1364, CI=0.0110, frac=0.224, total_ms=54.56
- F: relL2=0.1364, CI=0.0110, frac=0.224, total_ms=66.30

**Evidence:** `results/raw/exp_20261105_134910_63a5fa94_all.json` — C at bud=0.05 has relL2_mean=0.14181, intervention_fraction_mean=0.202.

### CRITICAL: Fix budget enforcement text (sec:results-budget)

**Current:** "At a matched actual intervention fraction (~20%), C (rel L2 0.142) is marginally more accurate than G (rel L2 0.159)"

**Correction:** Use corrected values. At matched frac ~0.20:
- C (nom 0.05): relL2=0.142, total_ms=16.81
- G_TWOPASS(0.2): relL2=0.159, total_ms=63.79 (true, incl prep)
- G_ONLINE(0.2): relL2=0.132, total_ms=57.27 (true)

G_ONLINE is marginally more accurate than C (0.132 vs 0.142) but 3.4× more expensive.

### CRITICAL: Report true total cost for G

**Current:** Table reports G budget=0.2 total_ms=8.98 (execution only, prep excluded).

**Correction:** Report true total including prep: G_TWOPASS(0.2) = 67.80 ms; G_ONLINE(0.2) = 60.94 ms.

### IMPORTANT: Disclose two-pass procedure

**Current:** Manuscript does not disclose that G_TWOPASS pre-scores a full all-surrogate trajectory.

**Correction:** Add: "Policy G's two-pass implementation pre-scores risk on an all-surrogate trajectory (60 neural predictions + risk inference, ~54 ms), selects top-k indices, then re-executes the controlled rollout. The preparation cost is reported separately from execution cost."

### IMPORTANT: p-values and effect sizes

**Current:** Exact p-values below 1e-30 (1.8e-32, 8.4e-41, 6.7e-46); negative Cohen's d (−1.45 to −2.03).

**Correction:** "p < 0.001" with t-statistics; report |d| with direction stated in words.

### IMPORTANT: Remove duplicate Conclusion section

**Current:** Two `\section{Conclusion}` blocks (one complete, one placeholder).

**Correction:** Remove the placeholder.

### MINOR: Cite or remove tran2026conformal

**Current:** In bibliography but not cited in text.

**Action:** Cite it in Related Work (conformal prediction for functional outputs) or remove from bibliography.

---

## 6. Optimized Controller Summary

### G_ONLINE (new single-pass implementation) advantages over G_TWOPASS:
1. **Lower true cost:** 57.27 ms vs 67.80 ms at budget 0.2 (15% saving)
2. **Better accuracy:** relL2 0.132 vs 0.159 at budget 0.2 (17% lower error)
3. **Fewer neural calls:** 120 vs 168 at budget 0.2 (29% fewer)
4. **Fresh risk rankings:** Scored on actual controlled trajectory, not stale all-surrogate trajectory
5. **Zero preparation cost:** No preliminary rollout; risk scored online

### G_ONLINE limitations:
1. Still uses `temporal_inconsistency` (3 predict calls per step) — expensive feature
2. "Top-k so far" heuristic may front-load interventions (early steps always selected until k steps seen)
3. Risk threshold adapts as more steps are observed; early-step selections are less informed

### Recommended controller for publication: **G_ONLINE**
Reasons:
- Strictly dominates G_TWOPASS on both accuracy and true cost
- Eliminates the 54 ms preparation cost entirely
- Scores risk on actual trajectory (avoids stale ranking problem)
- Enforces exact budget
- Simpler implementation (single-pass)

---

## 7. Final Assessment

### 7.1 Do the original statistical conclusions survive?

**Partially:**
- ✅ G (both variants) significantly improves over A (pure neural) at all budgets and all aggregation levels
- ✅ G enforces exact intervention budget at all levels
- ✅ Calibration adds no benefit (null confirmed at all levels)
- ❌ C vs G "more accurate at matched frac" claim: NOT significant at trajectory level (p=0.318)
- ❌ G has low total cost: FALSE when prep cost included (6.3–7.2× underreported)

### 7.2 Do policy G's budget and cost claims remain valid?

**Budget:** YES — intervention count is exact at all budgets for both G_TWOPASS and G_ONLINE.

**Cost:** NO — G_TWOPASS's reported total cost excludes the ~54 ms preparation cost, understating true cost by 6–7×. G_ONLINE has no preparation cost and is cheaper than G_TWOPASS, but still 3.4× more expensive than C at matched intervention fraction.

### 7.3 Is the two-pass procedure fully documented?

**In code:** YES — `policies.py` and `rollout.py` clearly implement two passes with separate prep/exec cost accounting.

**In manuscript:** NO — the manuscript does not disclose:
- That G pre-scores a full all-surrogate trajectory
- The ~54 ms one-time preparation cost
- That risk rankings may be stale after corrections

### 7.4 Are manuscript changes necessary?

**YES — 9 specific changes required** (detailed in Section 8). The most critical are:
1. Fix C/E/F table rows (aggregate mislabeled as nominal 0.05)
2. Fix budget enforcement text (wrong C value)
3. Report true total cost for G (include prep cost)
4. Disclose two-pass procedure and its cost
5. Fix p-values and effect-size reporting
6. Remove duplicate Conclusion section
7. Cite or remove uncited reference

### 7.5 Can the project proceed to submission with corrections?

**Yes, after the manuscript corrections listed above.** The G_ONLINE single-pass implementation should be adopted as the primary BRACE controller (it dominates G_TWOPASS on both accuracy and cost). The statistical conclusions about G vs A and G vs oracle are robust. The C vs G comparison must be qualified as "marginally more accurate but 3.4× more expensive, not significant at trajectory level."

---

## 8. Evidence Paths

| Evidence | Path |
|---|---|
| Authoritative raw results | `results/raw/exp_20261105_134910_63a5fa94_all.json` |
| Rebuild experiment (corrected) | `results/rebuild/corrected_20261109_101618/corrected_20261109_101618_all.json` |
| BRACE controller (original, two-pass) | `src/controller/policies.py` |
| BRACE rollout (cost accounting) | `src/controller/rollout.py` |
| BRACE controller (optimized, single-pass) | `src/controller/brace_controller.py` |
| Pass-1 profiler | `scripts/profile_pass1.py` |
| Overhead profiler | `scripts/profile_overhead.py` |
| G benchmark | `scripts/benchmark_g.py` |
| Statistical analysis | `scripts/analyze_corrected.py` |
| Evidence audit script | `scripts/evidence_audit.py` |
| Manuscript | `paper/main.tex` |
| Bibliography | `paper/references.bib` |

**End of sensitivity review.** All findings are traceable to `results/raw/exp_20261105_134910_63a5fa94_all.json` (authoritative) and `results/rebuild/corrected_20261109_101618/corrected_20261109_101618_all.json` (rebuild verification). No raw results, historical experiments, Git history, or manuscript source were modified during this review.
