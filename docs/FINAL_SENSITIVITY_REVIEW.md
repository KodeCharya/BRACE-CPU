# Final Statistical and Operational Methodology Sensitivity Review — BRACE-CPU

**Project directory:** `/home/mukesh/Documents/research/BRACE-CPU/`  
**Git branch:** `master`, commit `c8d3a9b`  
**Authoritative experiment:** `results/raw/exp_20261105_134910_63a5fa94_all.json` (3,500 records)  
**Review date:** 2026-11-05  
**Reviewer role:** Independent statistical and operational methodology reviewer  
**Rules followed:** Read-only inspection; no new experiments; no modifications to raw data, Git history, or manuscript source.

---

## 1. Statistical Dependence Assessment

### 1.1 Design structure

The corrected experiment uses a **crossed design**:
- **5 seeds** (0–4): independent model initializations and data shuffling
- **25 trajectories** per seed: the **same 25 trajectories** appear in all 5 seeds
- **125 (seed, trajectory) pairs**: each policy is evaluated on every pair
- **60 rollout steps** per trajectory: autocorrelated, never treated as independent in the analysis

The 3,500 records are **not** 3,500 independent samples. They are 5 seeds × 25 trajectories × 8 policies × up to 6 budget levels, with policies evaluated on identical test trajectories within each seed.

### 1.2 Levels of statistical aggregation

Three levels of aggregation were tested for paired comparisons (diff = policy_A − policy_B; negative = B more accurate):

| Comparison | Pair level (n=125) | Seed level (n=5) | Trajectory level (n=25) |
|---|---|---|---|
| G(0.2) vs A | diff=−0.0961, p<0.001, \|d\|=1.452 | diff=−0.0961, p=3.25e-06, \|d\|=16.47 | diff=−0.0961, p=2.20e-08, \|d\|=1.633 |
| G(0.4) vs A | diff=−0.1467, p<0.001, \|d\|=1.797 | diff=−0.1467, p=3.32e-06, \|d\|=16.38 | diff=−0.1467, p=4.61e-10, \|d\|=2.007 |
| G(0.6) vs A | diff=−0.1905, p<0.001, \|d\|=2.026 | diff=−0.1905, p=1.04e-06, \|d\|=21.92 | diff=−0.1905, p=1.09e-10, \|d\|=2.158 |
| C(nom 0.05) vs G(0.2) | diff=−0.0170, p=0.0408, \|d\|=0.185 | diff=−0.0170, p=0.0282, \|d\|=1.504 | diff=−0.0170, p=0.318, \|d\|=0.204 |
| H vs G(0.2) | diff=−0.1547, p<0.001, \|d\|=1.767 | diff=−0.1547, p=2.26e-05, \|d\|=10.12 | diff=−0.1547, p=1.01e-09, \|d\|=1.927 |
| F vs C at nom 0.05 | diff=−0.0054, p=0.292, \|d\|=0.095 | diff=−0.0054, p=0.134, \|d\|=0.838 | diff=−0.0054, p=0.574, \|d\|=0.114 |

### 1.3 Key statistical findings

**G vs A (the primary claim): ROBUST across all aggregation levels.**
- Pair level (n=125): p < 0.001
- Seed level (n=5, most conservative): p = 3.25e-06
- Trajectory level (n=25): p = 2.20e-08
- The conclusion that BRACE improves over pure neural is **not dependent** on treating 125 pairs as independent. Even with only 5 independent seeds, the result holds at p < 0.001.

**C vs G(0.2) (the marginal claim): NOT robust.**
- Pair level (n=125): p = 0.0408, nominally significant
- Seed level (n=5): p = 0.0282, still significant
- **Trajectory level (n=25): p = 0.318, NOT significant**
- The pair-level significance is driven by crossed dependence (multiple seeds evaluated on the same 25 trajectories). At the trajectory level — the level at which trajectories are independent — C is **not** significantly different from G.

**F vs C (calibration null): ROBUST across all levels.**
- Consistently non-significant at pair (p=0.292), seed (p=0.134), and trajectory (p=0.574) levels.

**Cluster bootstrap feasibility:**
- Two-way cluster bootstrap over seeds and trajectories is **not statistically appropriate** with only 5 seed clusters. Bootstrap variance estimation with 5 clusters is unreliable.
- The seed-level analysis (n=5) provides the most defensible conservative bound for between-seed generalization, but with only 4 degrees of freedom, confidence intervals are wide and should be interpreted cautiously.

### 1.4 Effect size sensitivity

| Comparison | Pair-level \|d\| | Seed-level \|d\| | Trajectory-level \|d\| |
|---|---|---|---|
| G(0.2) vs A | 1.452 | 16.466 | 1.633 |
| G(0.4) vs A | 1.797 | 16.376 | 2.007 |
| G(0.6) vs A | 2.026 | 21.921 | 2.158 |
| C vs G(0.2) | 0.185 | 1.504 | 0.204 |
| H vs G(0.2) | 1.767 | 10.124 | 1.927 |
| F vs C | 0.095 | 0.838 | 0.114 |

Cohen's d at the seed level is inflated because the seed-level standard deviation is very small (trajectories within a seed are highly correlated). The trajectory-level d values are more representative of the true effect size relative to trajectory-to-trajectory variability.

---

## 2. Sensitivity-Analysis Results

### 2.1 G vs A conclusion — SURVIVES all sensitivity checks

The conclusion that BRACE-CPU (G) improves long-horizon reliability over pure neural rollout (A) is robust:
- At every budget level (0.05, 0.1, 0.2, 0.4, 0.6), G has lower relL2 than A
- The difference is significant at p < 0.001 at pair, seed, and trajectory levels
- Effect sizes are large at all levels (trajectory-level \|d\| = 1.63–2.16)

### 2.2 C vs G conclusion — DOES NOT survive trajectory-level aggregation

The claim that "threshold policies are slightly more accurate than G at matched intervention fraction" relies on:
- The aggregate C value (relL2 = 0.049 across all budgets), NOT the value at matched intervention fraction
- Pair-level significance (p = 0.041) that disappears at the trajectory level (p = 0.318)

At the trajectory level (the correct unit for independent trajectories), C is **not** significantly different from G. The claim that C is "slightly more accurate" at matched intervention fraction is not supported by trajectory-level analysis.

### 2.3 CI width sensitivity

Confidence intervals at the pair level (n=125) are narrower than at the seed level (n=5) or trajectory level (n=25), as expected. For G(0.2) vs A:
- Pair CI95: [−0.1077, −0.0845], width = 0.0232
- Seed CI95: [−0.1012, −0.0910], width = 0.0102 (narrower due to averaging across 25 trajectories)
- Traj CI95: [−0.1192, −0.0730], width = 0.0462 (wider, reflecting true trajectory variability)

The trajectory-level CI is the most appropriate for generalizing to new trajectories.

---

## 3. BRACE Policy G Execution Sequence

### 3.1 Two-pass procedure (CONFIRMED)

Policy G (`BUDGET_RISK_RANKED`) uses a **two-pass procedure**:

**Pass 1 — Preparation (`prepare()` → `_prepare_budget_ranked()`):**
1. Start from initial state `u0`
2. For each of n_steps (60) steps:
   - Call `surrogate.predict(u, params, t)` → `u_pred` (neural inference #1 for this step)
   - Call `risk_estimator.score(surrogate, u, params, t, dt)` → risk score `r`
     - Internally calls `surrogate.predict()` AGAIN (neural inference #2 for this step)
     - Computes risk features and calls the logistic risk model
   - Store risk score `risks[i] = r`
   - Set `u = u_pred` (advance along all-surrogate trajectory)
3. Sort risks descending, select top-k indices where k = round(budget × n_steps)
4. Store `intervention_indices` (sorted)
5. **Total pass-1 cost: ~54 ms per rollout** (measured as `prepare_wall_s`)

**Pass 2 — Main rollout (`simulate_rollout` → `decide()`):**
For each step i from 0 to n_steps−1:
- Call `decide(u_t, params, t, dt, i)`:
  - For G: return `(i in intervention_indices, None, elapsed_lookup_time)` — no neural prediction, just a set membership lookup
- If `intervene` is True:
  - Call `solver.advance(u, t, dt, 1)` → one-step numerical correction from current state
  - Time as `solver_wall_s`
- If `intervene` is False:
  - `u_pred is None` (decide returned None for G)
  - Call `surrogate.predict(u, params, t)` → **SECOND neural prediction for this step**
  - Time as `neural_wall_s`
- Update `u`, advance time

### 3.2 Two-pass determination: YES, two-pass

Policy G explicitly uses a two-pass procedure:

| Pass | Purpose | Operations | Timed as |
|---|---|---|---|
| Pass 1 (prepare) | Risk pre-scoring and intervention selection | 60 neural predictions + 60 risk model inferences + ranking | `prepare_wall_s` (~54 ms, one-time) |
| Pass 2 (rollout) | Actual controlled execution | Neural prediction for non-intervened steps; solver call for intervened steps | `neural_wall_s` + `solver_wall_s` |

### 3.3 Risk rankings from uncorrected trajectory — applicability after correction

**The risk rankings are computed from an all-surrogate (uncorrected) trajectory.** Once numerical corrections are applied in pass 2, the state diverges from the pass-1 trajectory. The pre-computed risk rankings are therefore **not re-evaluated** after corrections.

This is a fundamental approximation: the controller decides which steps to correct based on risk scores computed along a trajectory that never receives corrections. After the first correction, the state `u` differs from what was used in pass-1 risk scoring, so subsequent risk rankings (which were fixed in pass 1) may no longer reflect the true risk at the actual state.

The code does not re-score risk after corrections. The intervention indices are fixed at the start of the rollout. This means:
- A step flagged as "high risk" in pass 1 may no longer be high risk after an earlier correction changes the state
- A step flagged as "low risk" in pass 1 may become high risk after an earlier correction

This is a known limitation of the pre-computation approach, but it is **not acknowledged** in the manuscript as a source of suboptimality.

---

## 4. Full Computational Cost Breakdown

### 4.1 Per-rollout cost components for BRACE-G (G)

From the raw records (averaged over 125 seed-traj pairs per budget):

| Budget | Interventions | Non-interventions | neural_wall_s (pass-2 non-int) | solver_wall_s | controller_wall_s | prepare_wall_s (pass-1) | total_wall_s REPORTED | total_wall_s TRUE (incl prep) |
|---|---|---|---|---|---|---|---|---|
| 0.05 | 3 | 57 | 7.99 ms | 0.57 ms | 0.11 ms | 53.94 ms | 8.67 ms | 62.61 ms |
| 0.1 | 6 | 54 | 7.58 ms | 1.12 ms | 0.12 ms | 53.77 ms | 8.82 ms | 62.59 ms |
| 0.2 | 12 | 48 | 6.75 ms | 2.21 ms | 0.13 ms | 54.02 ms | 9.09 ms | 63.11 ms |
| 0.4 | 24 | 36 | 5.07 ms | 4.41 ms | 0.15 ms | 53.84 ms | 9.63 ms | 63.47 ms |
| 0.6 | 36 | 24 | 3.41 ms | 6.58 ms | 0.16 ms | 53.96 ms | 10.15 ms | 64.10 ms |

### 4.2 Critical cost accounting finding

**`total_wall_s` does NOT include `prepare_wall_s`.**

Verification: For G at budget 0.2: neural (6.75) + solver (2.21) + controller (0.13) = 9.09 = total_wall_s. The prepare cost (54.02 ms) is recorded separately but **excluded** from the reported total.

This means:
- **Reported total cost understates true cost by 6.3×–7.2×** depending on budget
- **Prep cost is 84–86% of the true total cost** for G
- The "low online cost" of G (reported as ~9 ms at bud=0.2) is misleading — the true per-rollout cost is ~63 ms

### 4.3 Cost comparison at matched actual intervention fraction (~0.20)

| Policy | Budget param | Actual frac | relL2 | Reported total ms | True total ms (incl prep if applicable) |
|---|---|---|---|---|---|
| G (BRACE) | 0.2 | 0.200 | 0.159 | 9.09 | 63.11 |
| C (fixed threshold) | 0.05 | 0.202 | 0.142 | 15.62 | 15.62 |
| E (uncalibrated risk) | 0.05 | 0.224 | 0.136 | 50.85 | 50.85 |
| F (calibrated risk) | 0.05 | 0.224 | 0.136 | 61.61 | 61.61 |

**At matched actual intervention fraction (~0.20–0.22):**
- C is **more accurate** than G (0.142 vs 0.159 relL2) AND **4× cheaper** (15.62 vs 63.11 ms true cost)
- E is **more accurate** than G (0.136 vs 0.159) and comparable cost (50.85 vs 63.11 ms)
- F is **more accurate** than G (0.136 vs 0.159) and comparable cost (61.61 vs 63.11 ms)

**The claim that "BRACE-CPU's advantage is budget predictability plus low online cost" is incorrect on the cost dimension.** G's true cost is dominated by the one-time preparation pass, making it more expensive than C at matched intervention fraction while being less accurate.

### 4.4 Pareto-analysis implications

The Pareto figure (`pareto_accuracy_vs_intervention.png`) plots intervention fraction vs relL2. If total cost (including prep) were plotted instead of reported total:
- G's points would shift rightward by ~54 ms on the cost axis
- G would no longer appear on the lower-left (efficient) frontier
- C at bud=0.05 (relL2=0.142, cost=15.62 ms) would dominate G(0.2) (relL2=0.159, cost=63.11 ms)

The claim that G achieves "lower error than uncalibrated/fixed-threshold policies at matched intervention rates" is true for accuracy (0.159 vs 0.142 — actually C is better), but false for cost (G is 4x more expensive).

---

## 5. Execution Sequence Summary (BRACE-G)

```
PASS 1 — prepare() → _prepare_budget_ranked()
  u = u0.copy()
  for i in 0..59:
      u_pred = surrogate.predict(u, params, t)        # neural #1
      r = risk_estimator.score(surrogate, u, params, t, dt)  # neural #2 + risk model
      risks[i] = r
      u = u_pred                                      # advance along surrogate trajectory
  intervention_indices = top_k(risks)                  # sort + select
  Time: ~54 ms (one-time per rollout)

PASS 2 — simulate_rollout() loop
  for i in 0..59:
      intervene, u_pred, elapsed = decide(u, params, t, dt, i)
      # G: decide returns (i in indices, None, lookup_time) — NO neural prediction
      if intervene:
          u = solver.advance(u, t, dt, 1)              # numerical correction
      else:
          u = surrogate.predict(u, params, t)          # neural #3 (second for this step)
```

**Key point:** For G's non-intervened steps, the neural model is called **twice** per step — once in pass 1 (for risk scoring) and once in pass 2 (for actual state advancement). Only the pass-2 call is counted in `neural_wall_s`. The pass-1 calls are buried inside `prepare_wall_s` along with risk model inference and ranking overhead, so they cannot be separately quantified from the reported data.

---

## 6. Manuscript Changes Recommended

### 5.1 Table tab:summary — C/E/F rows mislabeled (CRITICAL)

**Current:** C/E/F rows show "budget=0.05 (nom.)" with relL2=0.049, frac=0.66 — these are **cross-budget aggregate values** (n=625), NOT values at actual budget 0.05.

**Corrected values at actual budget=0.05 (n=125):**
- C: relL2=0.1418, CI=0.0092, frac=0.202, total_ms=15.62
- E: relL2=0.1364, CI=0.0110, frac=0.224, total_ms=50.85
- F: relL2=0.1364, CI=0.0110, frac=0.224, total_ms=61.61

**Recommendation:** Either relabel rows as "all budgets (aggregate)" or report actual budget=0.05 values. Option (b) is fairer.

### 5.2 Budget enforcement section — incorrect C value

**Current:** "At a matched actual intervention fraction (~20%), threshold policies are slightly more accurate than G (rel L2 ≈ 0.049 vs 0.159)"

**Problem:** 0.049 is the cross-budget aggregate for C, not the value at matched frac. At actual matched frac (~0.20), C relL2 = 0.142.

**Correction:** "At a matched actual intervention fraction (~20%), C (nominal 5%, actual 20.2%, rel L2 0.142) is marginally more accurate than G (budget 20%, rel L2 0.159), but only because C intervenes ~3× more than its nominal budget."

### 5.3 Cost accounting — prepare cost excluded from total

**Current:** `total_wall_s = neural + solver + controller` (prepare excluded)

**Problem:** The ~54 ms one-time preparation cost is not included in the reported total cost, making G appear 6–7× cheaper than it truly is.

**Recommendation:** Either (a) include `prepare_wall_s` in `total_wall_s`, or (b) report `prepare_wall_s` prominently alongside total and clarify that it is a one-time per-rollout cost. The Pareto figure and any cost comparisons must use the true total cost.

### 5.4 p-values reported as exact below 1e-30

**Problem:** Values like 1.8e-32, 8.4e-41, 6.7e-46 are at double-precision underflow limits.

**Correction:** Report as "p < 0.001" with t-statistics.

### 5.5 Cohen's d sign convention

**Problem:** Negative d values (−1.45 to −2.03) without explaining that negative = G more accurate.

**Correction:** Report |d| with direction stated in words (e.g., "large effect favoring G, |d| = 1.45–2.03").

### 5.6 Duplicate Conclusion section

**Problem:** Two `\section{Conclusion}` blocks — one complete, one placeholder "[To be completed after main experiments.]".

**Correction:** Remove the placeholder.

### 5.7 Two-pass procedure not documented

**Problem:** The manuscript does not disclose that G runs a full all-surrogate trajectory (60 neural predictions + risk scoring) before the actual rollout, or that this one-time cost (~54 ms) dominates the per-rollout cost.

**Recommendation:** Add a clear explanation of the two-pass procedure and its cost implications.

### 5.8 Oracle (B) degeneracy not fully explained

**Problem:** B (always numerical) shows relL2 ≈ 0 because it uses the same solver/dt as the reference generator — it is numerically self-consistent, not a meaningful accuracy comparison.

**Correction:** Already noted in the table caption, but should be emphasized in the text.

---

## 7. Remaining Limitations

1. **Calibration null result:** Platt scaling provides no measurable benefit (ECE/Brier/AUROC identical, p=0.29). Reported honestly but undermines the "Risk-Calibrated" framing in the original title.

2. **OOD degradation:** Failure rate ~0.67 under unseen low-viscosity (ν=0.005–0.0079, below train range 0.01–0.03). AUROC degrades to ~0.90.

3. **Single PDE benchmark:** Burgers' equation only; advection-diffusion secondary.

4. **Small surrogate:** ~17K parameters; larger operators not evaluated.

5. **No comparison against ANCHOR/INC:** Different setting (lightweight CPU MLP vs neural operator), but ANCHOR is a direct predecessor that the manuscript acknowledges only in passing.

6. **Training nondeterminism:** PyTorch CPU training shows moderate run-to-run variation; mean±std across 5 seeds reported, bitwise reproducibility not claimed.

7. **Two-pass procedure cost not fully disclosed:** The ~54 ms one-time preparation cost is excluded from reported total cost.

8. **C vs G marginal claim not robust:** At trajectory level (n=25), p = 0.318 — not significant.

---

## 8. Final Assessment

### 8.1 Do the original statistical conclusions survive?

**Partially:**

| Claim | Survives? | Evidence |
|---|---|---|
| G vs A: BRACE improves over pure neural | **YES** — robust at pair, seed, and trajectory levels (p < 0.001 at all levels) | Pair: p<0.001; Seed: p=3.25e-06; Traj: p=2.20e-08 |
| G enforces budget exactly | **YES** — intervention-count budget is exact at all levels | frac = nominal at 0.05/0.1/0.2/0.4/0.6 |
| Calibration adds benefit | **NO** — null result confirmed at all levels | p = 0.292 (pair), p = 0.134 (seed), p = 0.574 (traj) |
| C more accurate than G at matched frac | **NOT ROBUST** — trajectory-level p = 0.318 | Pair: p=0.041; Seed: p=0.028; Traj: p=0.318 |
| G is cost-efficient | **NO** — true cost 6.3–7.2× higher than reported due to excluded prep cost | True total at bud=0.2: 63.11 ms vs reported 9.09 ms |

### 8.2 Do policy G's budget and cost claims remain valid?

**Budget enforcement:** YES — intervention-count budget is enforced exactly at all levels. The controller pre-selects top-k steps and intervenes only at those indices.

**Cost claims:** NO — as currently reported, G's cost is understated by 6.3–7.2× because the ~54 ms one-time preparation cost is excluded from `total_wall_s`. At true cost, G is **more expensive** than C/E/F at matched intervention fraction while being **less accurate** than C at matched fraction.

### 8.3 Is the two-pass procedure fully documented?

**PARTIALLY.** The code implements two passes:
- Pass 1: full all-surrogate trajectory for risk pre-scoring (~54 ms, one-time)
- Pass 2: actual controlled rollout with neural predictions for non-intervened steps

The code comments and docstrings describe this, but the **manuscript does not disclose**:
- That G runs a preliminary all-surrogate trajectory before the actual rollout
- That this preparation costs ~54 ms per rollout
- That this cost is excluded from reported total wall-clock time
- That risk rankings are computed from an uncorrected trajectory and may be stale after corrections

### 8.4 Are manuscript changes necessary?

**YES.** The following changes are necessary for accuracy:

1. **Fix C/E/F table rows** — show actual budget=0.05 values, not cross-budget aggregates
2. **Fix budget enforcement text** — use correct C value (0.142, not 0.049) at matched frac
3. **Include prepare cost in total** or report it prominently — do not let G appear 6–7× cheaper than it is
4. **Fix p-values** — report "p < 0.001" instead of exact sub-1e-30 values
5. **Fix Cohen's d sign convention** — explain that negative = G more accurate
6. **Remove duplicate Conclusion section**
7. **Add two-pass procedure disclosure** — explain pass-1 pre-scoring cost and its exclusion from reported total
8. **Clarify oracle degeneracy** — B uses same solver as reference, relL2 ≈ 0 is expected
9. **Cite or remove `tran2026conformal`** — currently in bibliography but not cited

---

## 9. Evidence References

| Evidence | Project-relative path |
|---|---|
| Authoritative raw results | `results/raw/exp_20261105_134910_63a5fa94_all.json` |
| Superseded defective run (excluded) | `results/raw/exp_20261105_132231_985e2bf7_all.json` |
| BRACE controller implementation | `src/controller/policies.py` |
| BRACE rollout implementation | `src/controller/rollout.py` |
| Risk estimator | `src/uncertainty/risk_estimator.py` |
| Risk features | `src/uncertainty/features.py` |
| Experiment runner | `scripts/run_experiment.py` |
| Statistical verification script | `scripts/stat_review.py` |
| Evidence audit script | `scripts/evidence_audit.py` |
| LaTeX syntax check | `scripts/latex_check.py` |
| Manuscript | `paper/main.tex` |
| Bibliography | `paper/references.bib` |
| Ablation output (calibration/OOD/feature) | `results/processed/ablation_output.txt` |
| Processed summary | `results/processed/summary_by_policy.csv` |
| Budget breakdown | `results/processed/G_brace_budget_by_budget.csv` |
| Dataset manifest | `data/processed/burgers/dataset_manifest.json` |
| Checkpoints | `checkpoints/surrogate_seed{0..4}.pt` |
| Expanded literature review | `docs/EXPANDED_LITERATURE_REVIEW.md` |
| Research positioning | `docs/FINAL_RESEARCH_POSITIONING.md` |
| Evidence audit | `docs/PUBLICATION_EVIDENCE_AUDIT.md` |
| Statistical audit | `docs/PUBLICATION_STATISTICAL_REVIEW.md` |
| Computational fairness audit | `docs/COMPUTATIONAL_FAIRNESS_AUDIT.md` |

---

**End of sensitivity review.** All findings are traceable to the corrected raw records in `results/raw/exp_20261105_134910_63a5fa94_all.json`. No raw results, historical experiments, Git history, or manuscript source were modified during this review.
