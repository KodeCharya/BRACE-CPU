# Computational Fairness Audit — BRACE-CPU

**Date:** 2026-11-05  
**Status:** CRITICAL DEFECT FOUND — must fix before manuscript finalization.

---

## 1. What is measured (per policy, per rollout)

In `src/controller/rollout.py::simulate_rollout`, the following are timed with `time.perf_counter()`:

| Quantity | Measured? | Where |
|---|---|---|
| Neural inference (surrogate.predict) | **Yes** | non-intervention steps |
| Numerical solver (solver.advance) | **Yes** | intervention steps |
| Total wall time | Yes | sum of neural + solver |
| Solver call count | Yes | `intervened.sum()` |

Each rollout reports `neural_wall_s`, `solver_wall_s`, `total_wall_s`, `solver_calls`, `intervention_fraction`.

## 2. CRITICAL DEFECT: risk/controller computation NOT timed

**Defect CF-1:** The time spent inside `Controller.decide()` — which includes the risk computation (feature extraction + risk-model forward + Platt scaling for policies E, F; residual computation for C, D; and G's one-time pass-1 preparation) — is **not measured or reported**.

Consequence: policies C, D, E, F, G appear cheaper than they truly are. Their `total_wall_s` undercounts actual cost by omitting controller/risk overhead. Only neural inference and solver time are counted.

**Defect CF-2 (related):** For risk-based policies (E, F, G), `decide()` calls `ctx.surrogate.predict(u_t, ...)` to obtain `u_pred` for the risk/residual computation, **and then** `simulate_rollout` calls `surrogate.predict(u, ...)` again on the non-intervention branch. So the neural model is evaluated **twice** per non-intervened step for policies E and F:
1. Once inside `decide()` (untimed, to compute risk/residual).
2. Once in the rollout `else` branch (timed as neural inference).

This means:
- The untimed prediction in `decide()` is "free" computation (not counted).
- The timed prediction in rollout duplicates work already done.

For policy G, `prepare()` runs an entire all-surrogate trajectory (n_steps predictions) to pre-score risk — this preparation cost is **not** included in `total_wall_s`.

## 3. Affected policies

| Policy | Untimed double compute? | Untimed prep cost? |
|---|---|---|
| A always_neural | No | No |
| B always_numerical | No | No |
| C fixed_threshold | **Yes** (residual needs u_pred) | No |
| D adaptive_threshold | **Yes** | No |
| E uncalibrated_risk | **Yes** (risk needs u_pred) | No |
| F calibrated_risk | **Yes** (risk needs u_pred) | No |
| G budget_risk_ranked | **Yes** (decide is just a lookup) | **Yes** (pass-1 all-surrogate trajectory, n_steps predictions) |
| H oracle | **Yes** (decide needs u_pred) | No |

## 4. Impact on reported results

- `total_wall_s` and `solver_wall_s` for risk-based policies are **underestimated**.
- Comparisons between A (no controller cost) and risk-based policies are **unfair** to A (A's cost is fully counted; risk-based policies' controller cost is not).
- The Pareto frontier (accuracy vs intervention fraction) is still valid for **intervention count**, but accuracy vs **wall-clock cost** is distorted for risk-based policies.

## 5. Required fix

Modify `simulate_rollout` / `Controller.decide` so that:

1. `decide()` returns both the intervention decision **and** the predicted next state `u_pred` (computed once), so the rollout reuses it instead of calling `predict` a second time.
2. The time spent in `decide()` (including feature extraction, risk model forward, Platt scaling) is measured with `perf_counter` and added to `controller_time_total`.
3. G's `prepare()` pass-1 time is measured and added to `prepare_time_total`.
4. `total_wall_s` = neural + solver + controller (+ prepare amortized or reported separately).
5. Report `controller_wall_s` and `prepare_wall_s` in the result record.

Until this fix is applied and experiments rerun, all wall-clock cost claims for risk-based policies (C–H) are **unreliable**.

## 6. One-time vs recurring costs (to be reported after fix)

- One-time (per experiment): surrogate training, risk-model fitting, Platt scaling, G's pass-1 preparation.
- Recurring (per rollout step): neural inference, risk computation, solver invocation.

These should be reported separately; training cost must not be amortized into per-rollout wall time without explicit statement.

---

*End of computational fairness audit. Fix required before final results are trusted.*
