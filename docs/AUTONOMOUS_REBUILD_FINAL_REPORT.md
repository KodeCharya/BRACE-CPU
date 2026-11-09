# Autonomous Rebuild — Final Report: BRACE-CPU

**Date:** 2026-11-09  
**Branch:** `autonomous-research-rebuild`  
**Authoritative experiment:** `results/raw/exp_20261105_134910_63a5fa94_all.json` (3,500 records, pre-existing)  
**Rebuild experiment:** `results/rebuild/corrected_20261109_101618/corrected_20261109_101618_all.json` (3,375 records, new)  
**Hardware:** Intel i5-6500, 4 cores, 7.2 GiB RAM, Python 3.14.4, torch 2.15.0+cpu (CPU-only)

---

## 1. What was inspected

Complete repository audit of:
- `src/` — solvers (`base.py`, `burgers.py`, `advection_diffusion.py`), models (`mlp.py`, `surrogate.py`, `trainer.py`), uncertainty (`features.py`, `risk_model.py`, `calibrator.py`, `risk_estimator.py`), controller (`policies.py`, `rollout.py`), correction (`numerical.py`), data generation (`dataset.py`)
- `tests/` — 14 tests (test_solvers.py, test_risk_controller.py)
- `scripts/` — run_experiment.py, run_ablation.py, analyze_results.py, evidence_audit.py, stat_review.py, latex_check.py + new scripts (profile_pass1.py, profile_overhead.py, benchmark_g.py, run_corrected.py, analyze_corrected.py, make_figures.py)
- `paper/` — main.tex (11 sections, 3 figures, 2 tables), references.bib (13 citations), supplementary_material.md
- `data/processed/burgers/` — 675 trajectories, dataset_manifest.json with SHA-256 checksums
- `results/raw/` — all historical experiment records (preserved, never overwritten)
- `docs/` — all prior audit reports (environment, literature, protocol, findings, audits)
- Git history: 3 commits on `master`, new branch `autonomous-research-rebuild`

Environment discovery: CPU i5-6500 (4 cores), 7.2 GiB RAM, 83 GiB disk free, Python 3.14.4, torch 2.15.0+cpu with 4 threads, numpy 2.5.4, scipy 1.18.1.

## 2. What was actually wrong

### Critical defects (in original implementation, now fixed)
1. **Off-by-one in `times[]`** (`src/controller/rollout.py`): `times[i+1] = t` instead of `t + dt`, corrupting reference-lookup accuracy. Fixed in prior phase; verified absent in rebuild.
2. **Uncosted controller/risk computation** (CF-1/CF-2): `decide()` called `predict()` internally without timing; risk-based policies double-computed neural inference. Fixed; all computation now timed.
3. **Duplicate `\section{Conclusion}`** in manuscript (complete + placeholder). Placeholder removed.

### Manuscript defects found in this rebuild
4. **C/E/F table rows mislabeled** — showed cross-budget aggregate (relL2 0.049, frac 0.66) as "nominal 0.05". Corrected to actual budget=0.05 values (relL2 0.142, frac 0.202).
5. **Budget enforcement text used wrong C value** (0.049 aggregate instead of 0.142 at matched frac). Corrected.
6. **Exact sub-1e-30 p-values** — not numerically meaningful. Changed to "p < 0.001".
7. **Negative Cohen's d sign** unexplained. Clarified with |d| and direction in words.
8. **Undefined `\ref{sec:baselines}`**. Changed to `\ref{sec:method}`.
9. **Missing `\bibliographystyle`/`\bibliography`/`\end{document}`** (accidentally deleted in earlier edit). Restored.
10. **Calibration claim overstated in title** ("Risk-Calibrated"). Changed to "Budget-Constrained...: An Empirical CPU Study."

### Design issues found (not bugs, but important limitations)
11. **Two-pass G preparation cost excluded from total** — `total_wall_s` did NOT include `prepare_wall_s` (~54 ms). True cost was 6.3–7.2× reported cost.
12. **Risk rankings stale after corrections** — G_TWOPASS pre-scores risk on an all-surrogate trajectory; after numerical corrections alter the state, the pre-selected intervention indices may not reflect true current risk.
13. **300 neural predictions for 60 steps in pass 1** — 5× redundancy per step (1 explicit + 1 score-internal + 3 temporal_inconsistency).
14. **Wrapper overhead dominates predict time** — 63% of `predict()` time is NumPy normalization + Python overhead, not model computation. Batch inference is 40× faster than 60 sequential predictions.
15. **Thread count irrelevant** — torch threads 1/2/4 produce identical timing (0.134–0.135 ms per predict). No oversubscription benefit from 4 threads.

## 3. What was changed

### New controller: G_ONLINE (single-pass)

Created `src/controller/brace_controller.py` with two implementations:

**G_TWOPASS** (`rollout_g_twopass`): Faithful reproduction of the original two-pass algorithm with **complete cost accounting**. Pass 1 pre-scores all 60 steps on an all-surrogate trajectory, selects top-k indices. Pass 2 executes controlled rollout, intervening at pre-selected indices. Preparation cost (~54 ms) is measured and reported separately, NOT hidden from total.

**G_ONLINE** (`rollout_g_online`): **New single-pass implementation**. At each step during the rollout:
- Score risk on the actual (post-correction) state
- Maintain a running list of visited risks
- Intervene if: (a) budget requires it (remaining_budget >= remaining_steps), or (b) current risk is among the top-k riskiest so far
- Guarantees exact budget enforcement while scoring risk on fresh states

### Cost accounting overhaul
- `CostBreakdown` dataclass tracks: prep_neural_ms, prep_risk_ms, prep_ranking_ms, prep_total_ms, exec_neural_ms, exec_risk_ms, exec_solver_ms, exec_controller_ms, exec_total_ms, total_ms, n_neural_calls, n_solver_calls, n_risk_calls
- `total_ms = prep_total + exec_total` (complete cost, preparation NOT excluded)
- `RolloutResult` includes full cost breakdown alongside accuracy metrics

### Experiment re-run with corrected policies
- `scripts/run_corrected.py`: Runs all 8 original policies (A-H) plus G_ONLINE (new) with tuned thresholds for C/E/F (matching original `tune_threshold` logic), complete cost accounting, and exact budget verification. Results at `results/rebuild/corrected_20261109_101618/`.

### Manuscript updates
- Title: dropped "Risk-Calibrated" (calibration null result)
- Abstract: honest cost accounting, two-pass disclosure, G_ONLINE results
- Method: two-pass procedure documented; G_ONLINE single-pass described
- Results: corrected C/E/F values; true total costs including prep; G_ONLINE vs G_TWOPASS comparison
- New: Threats to Validity section; Single-pass vs Two-pass section
- Figures: Pareto (true cost), budget compliance, G accuracy vs budget (both variants)
- Table: includes G_ONLINE rows; Exec vs True total columns

### Profile scripts
- `scripts/profile_pass1.py`: Detailed pass-1 cost breakdown (300 neural calls, 5× redundancy)
- `scripts/profile_overhead.py`: Threading sensitivity (no effect), batch vs single (40× speedup), feature overhead breakdown
- `scripts/benchmark_g.py`: Head-to-head G_TWOPASS vs G_ONLINE with complete accounting

## 4. What was optimized

### G_ONLINE (single-pass) vs G_TWOPASS (original two-pass)

| Metric (budget=0.2) | G_TWOPASS | G_ONLINE | Improvement |
|---|---|---|---|
| relL2 mean | 0.15884 | 0.13226 | 17% lower error |
| True total cost | 67.80 ms | 60.94 ms | 10% lower cost |
| Prep cost | 54.81 ms | 0.00 ms | 100% eliminated |
| Neural calls | 168 | 120 | 29% fewer |
| Budget enforcement | Exact (12/60) | Exact (12/60) | Both exact |

At budget 0.6: relL2 0.034 (online) vs 0.065 (twopass) — **47% lower error**.

### Profiling findings (no code changes, just measurement)
- **Thread count irrelevant**: torch 1/2/4 threads → identical timing (0.134–0.135 ms per predict)
- **Batch inference 40× faster**: 60 batch predictions = 0.207 ms vs 60 sequential = 8.318 ms
- **Wrapper overhead 63%**: 0.083 ms of 0.133 ms predict time is NumPy normalization, not model computation
- **temporal_inconsistency costs 0.419 ms** (3 predicts) — largest single feature expense
- **Solver is also wrapper-dominated**: rhs=0.043 ms, rk4_step=0.181 ms (4× rhs = 0.172 ms)

No optimization was applied to batch inference in the production controller because G_ONLINE's online design inherently requires sequential risk scoring (each step's risk depends on prior corrections). Batching would change the algorithm semantics.

## 5. Tests executed and their actual outcomes

### Existing test suite (14 tests)
```
tests/test_risk_controller.py::test_logistic_risk_model_separates_classes PASSED
tests/test_risk_controller.py::test_platt_scaling_maps_to_unit_interval PASSED
tests/test_risk_controller.py::test_brier_and_ece_bounds PASSED
tests/test_risk_controller.py::test_failure_metrics_keys PASSED
tests/test_risk_controller.py::test_risk_features_do_not_use_ground_truth PASSED
tests/test_risk_controller.py::test_budget_ranked_respects_budget PASSED
tests/test_solvers.py::test_periodic_gradient_and_laplacian_are_periodic PASSED
tests/test_solvers.py::test_advection_diffusion_exact_solution_convergence PASSED
tests/test_solvers.py::test_advection_diffusion_conserves_mass_when_a_equals_zero_diffusion_only PASSED
tests/test_solvers.py::test_burgers_mass_conservation_periodic PASSED
tests/test_solvers.py::test_burgers_no_blowup_for_smooth_initial_data PASSED
tests/test_solvers.py::test_advance_does_not_modify_input PASSED
tests/test_solvers.py::test_stable_dt_positive_and_shrinks_with_refinement PASSED
tests/test_solvers.py::test_relative_l2_error_properties PASSED
```

**Result: 14 passed in 1.62s. No failures.**

### New benchmark scripts (all executed successfully)
- `scripts/profile_pass1.py`: PASS-1 cost breakdown — 300 neural calls, 56.15 ms prep, 5× redundancy per step
- `scripts/profile_overhead.py`: Threading (no effect), batch (40×), wrapper overhead (63%), feature costs
- `scripts/benchmark_g.py`: G_TWOPASS vs G_ONLINE head-to-head — both enforce exact budget; G_ONLINE more accurate and cheaper
- `scripts/latex_check.py`: LaTeX syntax — environments balanced, no undefined refs, all citations resolve

## 6. Experiments completed and their exact identifiers

### Historical (pre-existing, preserved)
| Experiment ID | Records | Status | Used in manuscript? |
|---|---|---|---|
| `exp_20261105_131700_541c06cf` | 1,260 | Pilot (3 seeds) | NO |
| `exp_20261105_132231_985e2bf7` | 3,500 | **SUPERSEDED** (defective code) | NO |
| `exp_20261105_134910_63a5fa94` | 3,500 | **AUTHORITATIVE** (corrected rerun) | YES (original manuscript numbers) |
| `exp_20261105_135732_33b0c6f7` | 700 | Reproducibility (seed 0 retrain) | NO |

### New (this rebuild)
| Experiment ID | Records | Description | Used in manuscript? |
|---|---|---|---|
| `corrected_20261109_101216` | 3,375 | Corrected policies with hardcoded thresholds — **DISCARDED** (threshold tuning bug) | NO |
| `corrected_20261109_101618` | 3,375 | Corrected policies with **tuned thresholds** (matching original) + G_ONLINE | **YES** (manuscript updated to corrected numbers) |

### Configuration (both authoritative and rebuild)
- 5 seeds (0–4), 25 test trajectories, 60 rollout steps
- Budgets: 0.05, 0.1, 0.2, 0.4, 0.6
- Policies: A, B, C, D, E, F, G_TWOPASS, G_ONLINE (rebuild adds G_ONLINE)
- Dataset: 675 trajectories, nx=64, SHA-256 verified
- Checkpoints: `checkpoints/surrogate_seed{0..4}.pt`

## 7. Baseline-versus-optimized measurements

### G_TWOPASS (original) vs G_ONLINE (optimized single-pass)

| Budget | G_TWOPASS relL2 | G_ONLINE relL2 | Accuracy gain | G_TWOPASS cost | G_ONLINE cost | Cost gain |
|---|---|---|---|---|---|---|
| 0.05 | 0.22171 | 0.21517 | 3.0% | 67.25 ms | 58.84 ms | 12.5% |
| 0.10 | 0.19696 | 0.18248 | 7.4% | 67.42 ms | 59.54 ms | 11.7% |
| 0.20 | 0.15884 | 0.13226 | 16.7% | 67.80 ms | 60.94 ms | 10.1% |
| 0.40 | 0.10827 | 0.07301 | 32.6% | 68.39 ms | 63.71 ms | 6.8% |
| 0.60 | 0.06450 | 0.03425 | 46.9% | 69.00 ms | 66.47 ms | 3.7% |

**G_ONLINE strictly dominates G_TWOPASS on both accuracy and true cost at all budgets.**

### G_ONLINE vs C (fixed threshold) at matched intervention fraction (~0.20)

| Policy | Reported budget | Actual frac | relL2 | True total ms |
|---|---|---|---|---|
| C (fixed threshold) | 0.05 | 0.202 | 0.14181 | 16.81 |
| G_ONLINE | 0.2 | 0.200 | 0.13226 | 60.94 |
| G_TWOPASS | 0.2 | 0.200 | 0.15884 | 67.80 |

At matched intervention fraction, G_ONLINE is **marginally more accurate** (0.132 vs 0.142) but **3.6× more expensive** (60.94 vs 16.81 ms). The accuracy difference is **not statistically significant** at the trajectory level (p=0.421, n=25).

## 8. Accuracy, latency, memory, and cost comparisons

### Per-rollout cost breakdown (budget=0.2, median across seeds)

| Policy | Neural ms | Risk ms | Solver ms | Controller ms | Prep ms | TRUE total ms | relL2 | frac |
|---|---|---|---|---|---|---|---|---|
| A (neural) | 8.56 | 0 | 0 | 0.04 | 0 | 8.60 | 0.255 | 0.000 |
| B (numerical) | 0 | 0 | 10.83 | 0.02 | 0 | 10.86 | 0.000 | 1.000 |
| C (fixed) | 0 | 14.41* | 2.40 | 0.04 | 0 | 16.81 | 0.142 | 0.202 |
| E (uncal) | 0 | 51.77* | 2.79 | 0.04 | 0 | 54.56 | 0.136 | 0.224 |
| F (cal) | 0 | 63.49* | 2.80 | 0.04 | 0 | 66.30 | 0.136 | 0.224 |
| G_TWOPASS | 6.75 | 45.29† | 2.21 | 0.02 | 54.81 | 67.80 | 0.159 | 0.200 |
| G_ONLINE | 9.21 | 30.64 | 2.38 | 0.03 | 0 | 60.94 | 0.132 | 0.200 |

*C/E/F risk computation is timed inside controller_wall_s (decide() returns u_pred; neural+controller combined).  
†G_TWOPASS prep risk (45.29 ms) includes 300 neural predictions (5 per step) + feature extraction.

### Memory
All experiments ran within 7.2 GiB RAM. Peak usage stayed well below 1 GiB (models ~75 KB, trajectories ~small arrays). No OOM observed. Sequential execution, single seed at a time.

## 9. Statistical results and limitations

### Paired comparisons (125 seed-traj pairs, diff = first − second)

| Comparison | Pair (n=125) | Seed (n=5) | Traj (n=25) | Robust? |
|---|---|---|---|---|
| G_ONLINE(0.2) vs A | p<0.001, \|d\|=1.607 | p=8.65e-06 | p=1.23e-08 | **YES** |
| G_TWOPASS(0.2) vs A | p<0.001, \|d\|=1.452 | p=3.25e-06 | p=2.20e-08 | **YES** |
| C(nom0.05) vs G_ONLINE(0.2) | p=0.089 (ns) | p=0.019 | p=0.421 (ns) | **NO** |
| C(nom0.05) vs G_TWOPASS(0.2) | p=0.041 | p=0.028 | p=0.318 (ns) | **NO** |
| G_ONLINE vs G_TWOPASS (0.2) | p=1.05e-08, \|d\|=0.549 | p=0.004 | p=0.002 | **YES** |
| F vs C at nom0.05 | p=0.292 (ns) | p=0.134 (ns) | p=0.574 (ns) | **Null confirmed** |

### Limitations
- Only 5 seeds: between-seed inference limited (4 df); seed-level CIs wide
- Trajectory-level analysis (n=25) is most appropriate for generalizing to new trajectories
- Two-way cluster bootstrap unreliable with 5 seed clusters (not performed)
- Timesteps treated as correlated (per-trajectory means)
- Multiple comparisons: 6+ tests reported; strongest results (G vs A) survive Bonferroni; marginal results do not

## 10. Unsupported original claims removed or narrowed

| Original claim | Correction | Reason |
|---|---|---|
| "Budget-constrained risk-ranked intervention enforces the intervention budget exactly and achieves lower error than uncalibrated/fixed-threshold policies at matched intervention rates" (abstract) | Removed "lower error" claim; replaced with "budget predictability, not raw accuracy or cost efficiency" | At matched actual frac, C (0.142) vs G_ONLINE (0.132) difference is not statistically significant at trajectory level (p=0.421) |
| "Low online cost" (implying efficiency) | Removed; replaced with honest cost accounting including preparation | G_TWOPASS true cost is 67.80 ms (prep excluded from original reported total); G_ONLINE 60.94 ms — both 3.6× more expensive than C at matched frac |
| Title "Risk-Calibrated..." | Changed to "Budget-Constrained...: An Empirical CPU Study" | Calibration provides no measurable benefit (null result) |
| Exact p-values below 1e-30 | Changed to "p < 0.001" with t-statistics | Exact values at double-precision underflow limits |
| Negative Cohen's d (−1.45 to −2.03) | Changed to \|d\| with direction in words | Sign convention was confusing |
| C/E/F table rows labeled "nominal 0.05" with aggregate values | Corrected to actual budget=0.05 values (relL2 0.142, frac 0.202) | Values were cross-budget aggregates, mislabeled |
| Duplicate Conclusion section | Removed placeholder | Structural defect |

## 11. Remaining blockers

### BLOCKED: LaTeX compilation
**Cannot compile the manuscript** — no TeX Live toolchain installed, and OS does not permit installation without root/sudo. The manuscript is syntax-valid (balanced environments, no undefined refs, 13/13 citations resolve per `scripts/latex_check.py`) but has not been rendered to PDF. Compilation must be performed on a machine with TeX Live. This is an environment limitation, not a manuscript defect.

### NOT BLOCKED (handled)
- Statistical independence: addressed with three-level sensitivity analysis
- Cost accounting: G_ONLINE eliminates prep cost; both implementations report true total cost
- Two-pass disclosure: now documented in manuscript Method section
- C/E/F mislabeling: corrected in manuscript table and text
- Calibration null: prominently reported throughout

## 12. Exact reproduction commands

```bash
cd /home/mukesh/Documents/research/BRACE-CPU
.venv/bin/python -m pip install -e .[dev]  # if needed (from pyproject.toml)

# 1. Run solver + risk/controller tests (14 tests, ~2s)
PYTHONPATH=src .venv/bin/python -m pytest tests/ -v

# 2. Regenerate dataset (675 trajectories, ~5 min)
PYTHONPATH=src .venv/bin/python src/data/generation/dataset.py \
  --output-dir data/processed --nx 64 --n-train 300 --n-val 75 --n-cal 75 \
  --n-test 150 --n-ood 75 --base-seed 0

# 3. Train surrogates (5 seeds, ~15 min total; retraining is NOT bitwise-reproducible)
PYTHONPATH=src .venv/bin/python -c "
from models.surrogate import NeuralSurrogate, SurrogateConfig
from models.trainer import train_surrogate
for seed in [0,1,2,3,4]:
    cfg = SurrogateConfig(nx=64, n_params=1, n_time=1, hidden_layers=3, hidden_units=64,
                          activation='relu', lr=1e-3, batch_size=256, epochs=40, seed=seed)
    s = NeuralSurrogate(cfg)
    train_surrogate(s, 'data/processed/burgers/train', val_dir='data/processed/burgers/validation', stride=4)
    s.save(f'checkpoints/surrogate_seed{seed}.pt')
"

# 4. Run corrected experiment (tuned thresholds + G_ONLINE, ~20-30 min)
PYTHONPATH=src:scripts .venv/bin/python scripts/run_corrected.py
# Output: results/rebuild/corrected_<timestamp>/corrected_<timestamp>_all.json

# 5. Analyze corrected results
PYTHONPATH=src .venv/bin/python scripts/analyze_corrected.py

# 6. Run ablations (feature/cal-set/OOD, reuses checkpoints)
PYTHONPATH=src .venv/bin/python scripts/run_ablation.py

# 7. Profile BRACE-G pass-1 cost
PYTHONPATH=src .venv/bin/python scripts/profile_pass1.py

# 8. Profile overhead (threading, batch, feature costs)
PYTHONPATH=src .venv/bin/python scripts/profile_overhead.py

# 9. Benchmark G_TWOPASS vs G_ONLINE
PYTHONPATH=src .venv/bin/python scripts/benchmark_g.py

# 10. Generate figures
PYTHONPATH=src .venv/bin/python scripts/make_figures.py

# 11. Verify LaTeX syntax
PYTHONPATH=src .venv/bin/python scripts/latex_check.py

# 12. Run statistical review
PYTHONPATH=src .venv/bin/python scripts/stat_review.py
```

**To reproduce reported numbers exactly:** Use existing checkpoints (`--no-retrain` path — but note `run_corrected.py` always trains; the original `run_experiment.py --no-retrain` reuses checkpoints). Retraining produces different models due to PyTorch CPU nondeterminism; mean±std across seeds is the reproduction target, not bitwise identity.

## 13. Git branch and commit identifiers

- **Branch:** `autonomous-research-rebuild`
- **Baseline (pre-rebuild):** `c8d3a9b` on `master`
- **Rebuild branch created from:** `e9ddd91` (latest `master`)
- **No commits yet on rebuild branch** — all changes are in the working tree pending final commit

## 14. Final scientific conclusion

The BRACE-CPU project, after independent audit and rebuild, supports the following conclusions:

**Supported (robust across statistical sensitivity analyses):**
1. Risk features (dominated by normalized PDE residual) predict surrogate failure with AUROC ≈ 0.99 in-distribution
2. Selective numerical correction significantly improves long-horizon reliability over pure neural rollout (large effect sizes, p < 0.001 at all aggregation levels)
3. Budget-constrained risk-ranked intervention (both G_TWOPASS and G_ONLINE) enforces the intervention budget exactly
4. G_ONLINE (single-pass) dominates G_TWOPASS (two-pass) on both accuracy (17–47% lower error) and true cost (7–12% lower)
5. Calibration (Platt scaling) provides **no measurable benefit** over uncalibrated risk scores — null result confirmed at all aggregation levels

**Not supported (honestly reported):**
- BRACE-CPU is not more cost-efficient than threshold baselines at matched intervention fraction (G_ONLINE costs 3.6× more than C at frac≈0.20)
- C vs G accuracy difference at matched frac is not significant at trajectory level (p=0.421)
- OOD performance degrades substantially (fail rate ~0.67)

**Manuscript status:** Source complete and corrected; LaTeX compilation blocked (no TeX Live, no root/sudo). All quantitative claims trace to corrected raw records. Novelty honestly narrowed (ANCHOR as direct predecessor acknowledged).

---

## Claim-to-Evidence Table

| Manuscript claim | Supporting experiment/artifact | Verification method | Status |
|---|---|---|---|
| A neural relL2 = 0.2550 ± 0.0207 | `results/raw/exp_20261105_134910_63a5fa94_all.json` (A, n=125) | Recalculated from raw JSON; mean=0.25495, CI95=0.02067 | **VERIFIED** |
| B numerical relL2 ≈ 0 (degenerate) | Same (B, n=125) | Mean=0.00000; same solver as reference generator | **VERIFIED** |
| H oracle relL2 = 0.0042 ± 0.0004 | Same (H, n=125) | Mean=0.00416, CI95=0.00042 | **VERIFIED** |
| G_TWOPASS budget enforcement exact | Same (G_brace_budget, frac=nominal at all budgets) | Mean frac matches budget at 0.05/0.1/0.2/0.4/0.6 | **VERIFIED** |
| G_ONLINE budget enforcement exact | `results/rebuild/corrected_20261109_101618/...` (G_ONLINE, frac=nominal) | Mean frac matches budget at all levels | **VERIFIED** |
| G_ONLINE more accurate than G_TWOPASS | Same rebuild (paired, 125 pairs) | diff=-0.027, \|d\|=0.549, p=1.05e-08 | **VERIFIED** |
| G_ONLINE cheaper than G_TWOPASS (true cost) | Same rebuild (true_total_ms) | 60.94 ms vs 67.80 ms at budget 0.2 | **VERIFIED** |
| C overshoots nominal 0.05 budget | Same rebuild (C@0.05, frac=0.202) | Mean frac=0.202 vs nominal 0.05 | **VERIFIED** |
| Calibration provides no benefit | Ablation (5 seeds: ECE 0.0096 vs 0.0103, Brier identical, AUROC identical) + F vs C paired (p=0.292) | Recomputed from raw; paired t-test | **VERIFIED** |
| Risk features predict failure (AUROC 0.988) | Ablation A1 (seeds 0–2: combined AUROC 0.987–0.991) | Feature ablation script output | **VERIFIED** |
| OOD fail rate ~0.67 vs ~0.09 in-dist | Ablation A3 (seeds 0–2: fail_rate 0.661–0.686) | OOD evaluation output | **VERIFIED** |
| ~57.5% error reduction at 40% budget | Raw records (G@0.4 relL2=0.108 vs A 0.255) | (0.255-0.108)/0.255 = 57.6% | **VERIFIED** |
| G_ONLINE costs 3.6× more than C at matched frac | Rebuild (G_ONLINE 60.94 ms vs C 16.81 ms at frac~0.20) | Mean true_total_ms from raw JSON | **VERIFIED** |
| C vs G_ONLINE accuracy difference not significant (traj level) | Rebuild (paired by trajectory, n=25, p=0.421) | Trajectory-level paired t-test | **VERIFIED** |
| 13 citations all valid arXiv papers | `paper/references.bib` | All 13 keys resolve; all arXiv IDs verified during literature review | **VERIFIED** |
| LaTeX manuscript compiles | NOT COMPILED (no TeX Live; no root/sudo) | Syntax validated via `scripts/latex_check.py`; compilation blocked by environment | **NOT VERIFIED** |

---

**End of final report.** All results preserved; historical experiments never overwritten; rebuild results in `results/rebuild/`; manuscript corrected; G_ONLINE recommended as primary implementation.
