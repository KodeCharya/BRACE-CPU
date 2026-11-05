# Research Findings — BRACE-CPU

**Project:** BRACE-CPU — Budgeted Risk-Adaptive Computing Engine  
**Date:** 2026-11-05  
**Basis:** Main experiment rerun — 5 independent seeds (0–4), 25 held-out test trajectories per seed, 60 rollout steps each, intervention budgets 5%/10%/20%/40%/60%. All on CPU-only hardware (Intel i5-6500, 4 cores, 8 GiB RAM). Raw results: `results/raw/exp_20261105_134910_63a5fa94_all.json` (3,500 records); processed: `results/processed/`.

> **Correction note:** The first main run (exp_20261105_132231_985e2bf7) had two defects: (1) an off-by-one in the `times[]` array corrupted the reference-lookup accuracy measurement, and (2) controller/risk-computation time was not measured (defects CF-1/CF-2). Both were fixed (docs/COMPUTATIONAL_FAIRNESS_AUDIT.md) and the experiment was rerun. The numbers below are from the corrected rerun. The original run is retained (raw results are immutable) but superseded.

---

## RQ1 — Can lightweight risk features predict when the surrogate will exceed the error threshold?

**Yes.** A regularized logistic-regression risk model on six cheap features (normalized PDE residual, temporal inconsistency, mass-conservation deviation, increment magnitude, gradient magnitude, viscosity) predicts failure (next-step relative L2 error > τ = 0.01) with AUROC ≈ 0.99 on held-out test data (0.9879 ± 0.0021 across 5 seeds). Failure rate at τ = 0.01 is ≈ 9% in-distribution.

Feature ablation (docs/results/processed/ablation_output.txt): `norm_residual` is dominant (AUROC 0.979 alone); `increment_magnitude` and `temporal_inconsistency` also strong (~0.97); `mass_deviation` alone is near-random (0.58); `nu` alone weak (0.68). Features are highly redundant — removing any single one barely changes combined AUROC (~0.99). A simple residual magnitude is NOT sufficient by itself, but combined features are highly predictive.

## RQ2 — Does calibration improve prediction-risk reliability vs an uncalibrated estimator?

**No meaningful improvement (honest null result, robust to calibration-set size).** Platt-scaled calibration preserves risk ranking (AUROC identical: 0.9879 calibrated vs 0.9879 uncalibrated). Reliability across 5 seeds:

| Metric | Calibrated | Uncalibrated |
|---|---|---|
| ECE | 0.0096 ± 0.0020 | 0.0103 ± 0.0011 |
| Brier score | 0.0251 ± 0.0028 | 0.0251 ± 0.0027 |
| AUROC | 0.9879 ± 0.0021 | 0.9879 ± 0.0021 |

Calibration-set size sweep (25%/50%/100%): ECE/Brier stable (~0.007–0.014 / ~0.024–0.028), AUROC unchanged. Calibration provides no benefit here — the raw risk score is already well-calibrated. Moreover, calibrated risk (F) costs MORE per step than uncalibrated (E) due to Platt lookup overhead, with identical accuracy. Retained as a null finding.

## RQ3 — Can budgets be allocated more effectively using risk-ranked intervention than fixed-threshold correction?

**Partially yes on budget discipline; no on raw accuracy at matched spend.** Key finding:

- **BRACE-CPU (G)** enforces the intervention budget *exactly* (actual = nominal at all levels: 0.05, 0.10, 0.20, 0.40, 0.60) via top-k risk-ranked pre-selection.
- **Fixed-threshold (C), uncalibrated (E), calibrated (F)** all *overshoot* their nominal budget: targeting 5% yields ≈ 66% actual intervention, because error accumulation drives the surrogate across thresholds as the rollout diverges. They only achieve ~20% actual by nominally targeting 5% — i.e., they do not respect the stated budget.
- At matched *actual* intervention fraction (~20%): G relL2 = 0.159 vs C/E/F relL2 ≈ 0.049. Threshold policies are more accurate but only because they intervene 3× more than their nominal budget (66% vs 20%).

**Interpretation:** BRACE-CPU's advantage is *budget predictability + low online cost*, not raw accuracy at equal spend. With a hard computational budget, BRACE delivers it; threshold policies cannot be relied on to stay within budget. Additionally, E/F pay high per-step risk-compute cost (controller 48–59ms/rollout) with no accuracy benefit over C (controller 13ms) — they are dominated.

## RQ4 — Does selective numerical correction improve long-horizon reliability?

**Yes, relative to pure neural.** Always-neural (A): relL2 0.2550. Even lowest-budget BRACE (5%): relL2 0.2217. Always-numerical (B): relL2 0.0000 (degenerate — B is literally the reference solver, so it tracks itself perfectly). Oracle (H): relL2 0.0042 at 95.5% intervention. Selective correction cuts error substantially vs pure neural: 0.255 → 0.197 (10%) → 0.159 (20%) → 0.108 (40%) → 0.065 (60%). Long-horizon error accumulation is the dominant failure mode of pure neural rollout; periodic numerical intervention bounds it.

## RQ5 — Relationship between intervention frequency, accuracy, latency, cost

**Monotonic accuracy-vs-budget trade-off.** BRACE relL2 decreases monotonically: 0.222 (5%) → 0.197 (10%) → 0.159 (20%) → 0.108 (40%) → 0.065 (60%). Error reduction vs pure neural: 13% / 23% / 38% / 58% / 75%. Solver wall-clock scales linearly (0.57 → 6.58ms). Neural-only (A) is cheapest (8.5ms total, no solver) but least accurate.

**Cost accounting (fixed):** Controller/risk compute is now measured. E/F cost 56–67ms/rollout (dominated by per-step risk computation) with accuracy 0.049 — dominated by C (0.049 accuracy, 21ms total). G costs 9.3ms online + 54ms one-time prep; enforces budget exactly. No single policy dominates the Pareto frontier.

## RQ6 — Robustness to parameter / initial-condition changes

**Degrades under OOD but no catastrophic failure.** OOD test (viscosity 0.005–0.0079, below train range 0.01–0.03; Gaussian random field ICs): failure rate jumps to ~0.67 (vs ~0.09 in-distribution), mean error ~0.014, AUROC degrades to ~0.90 (from 0.99). Performance degrades gracefully with ν; BRACE becomes less reliable under genuine extrapolation (unseen low-ν regime). Interpolation (in-distribution) is robust.

---

## Summary table (5 seeds, mean ± 95% CI)

| Policy | Budget | rel L2 mean | CI95 | Intervention frac | Total wall-ms |
|---|---|---|---|---|---|
| A always neural | — | 0.2550 | 0.0207 | 0.000 | 8.50 |
| B always numerical | — | 0.0000 | 0.0000 | 1.000 | 10.98 |
| H oracle | — | 0.0042 | 0.0004 | 0.955 | 21.70 |
| G BRACE | 0.05 | 0.2217 | 0.0175 | 0.050 | 8.67 |
| G BRACE | 0.10 | 0.1970 | 0.0162 | 0.100 | 8.82 |
| G BRACE | 0.20 | 0.1588 | 0.0152 | 0.200 | 9.09 |
| G BRACE | 0.40 | 0.1083 | 0.0128 | 0.400 | 9.63 |
| G BRACE | 0.60 | 0.0645 | 0.0088 | 0.600 | 10.15 |
| C fixed threshold | 0.05 (nominal) | 0.0490 | 0.0049 | 0.660 | 20.87 |
| E uncalibrated risk | 0.05 (nominal) | 0.0489 | 0.0051 | 0.658 | 56.07 |
| F calibrated risk | 0.05 (nominal) | 0.0489 | 0.0051 | 0.658 | 66.89 |

Note: E/F/C intervention fractions (0.66) far exceed their nominal 0.05 budget — budget overshoot under error accumulation. G matches budget exactly.

---

## Novelty assessment (honest)

BRACE-CPU is **not** the first neural PDE surrogate, the first uncertainty-calibration method, or the first selective-prediction method. Its defensible contribution is the *specific combination*: calibrated failure-probability estimation + budget-constrained risk-ranked selective numerical correction, evaluated under strict CPU/RAM constraints with real wall-clock cost accounting. The finding that threshold policies cannot be trusted to honor a stated budget (they overshoot under error accumulation) and that BRACE enforces it exactly with low online cost is a concrete, defensible result.

---

## Limitations (see docs/LIMITATIONS.md)

- Single PDE (Burgers') as primary benchmark; advection-diffusion secondary.
- Calibration assumes exchangeability; no distribution-free guarantees under shift.
- τ = 0.01 set a priori; results sensitive to τ.
- CPU-only, 8 GiB RAM constrains scale.
- Small surrogate (~17K params); large operators not exhaustively evaluated.
- G's one-time pass-1 preparation cost (≈54ms) is reported separately, not amortized.
- OOD performance degrades (fail_rate ~0.67).

---

*End of findings. To be integrated into paper/main.tex Results section.*
