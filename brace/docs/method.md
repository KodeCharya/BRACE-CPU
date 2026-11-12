# Method

## 1. Setting

Let `u_t` be the state of a PDE on a periodic spatial grid at saved time index
`t`, obtained from a reference solver advancing one saved interval per index.
A neural surrogate ensemble `{g_m}` predicts the increment

```
u_{t+1} = u_t + g_m(u_t) / M  (averaged over members)
```

so the identity path is exact and the network only has to model the local
increment. Rollouts are autoregressive: the surrogate consumes its own output.

Running the reference solver for the whole horizon costs `T` saved intervals.
A correction at step `t` integrates the same governing PDE from the current
(possibly surrogate) state over exactly one saved interval, costing 1 unit.
A budget `B` is a fraction of the full-solver cost, so a step-correction policy
may act `k = floor(B*T)` times per trajectory. This makes `B = 1` directly
comparable to "just run the solver".

## 2. Online risk features

At each step, before committing to the surrogate prediction, nine features are
computed from quantities available online (ensemble spread, PDE residual,
updates, history, time):

```
ens_std_mean, ens_std_max, residual_rel, conservation_dev, update_mag,
temporal_inconsistency, hist_residual, hist_std, time_frac
```

`residual_rel` comes from the discrete PDE operator applied to the ensemble
mean; `temporal_inconsistency` compares the surrogate increment with the PDE's
predicted increment; history terms are exponentially weighted running means.

## 3. Risk models

Two ridge/logistic models are fit **on validation trajectories only**:

- a *step* model: expected one-step relative error and failure probability
  (`error > tau`), used for `ranked_calibrated`;
- a *horizon* model: discounted future error
  `sum_{s>=t} gamma^{s-t} e_s` with `gamma = 0.95`, used for `brace`.

Split conformal calibration on the validation residuals produces a
distribution-free interval for the one-step error at nominal coverage
`1 - alpha` (`alpha = 0.1`). Exchangeability is assumed, not guaranteed; we
measure empirical coverage on shifted splits rather than claiming validity.

`brace_combo` fuses the standardised horizon score with standardised ensemble
disagreement using fixed equal weights (chosen a priori, not tuned on test).

## 4. Gain model

A correction's benefit is not the same as the risk of a step. For each
validation step we compute the *counterfactual* gain

```
gain = e_neural - e_corrected
```

where `e_corrected` is the one-step error after integrating the reference solver
from the uncorrected surrogate state. We fit a ridge model of the **relative**
gain `gain / e_neural`, clipped to `[0,1]`, on the nine online features. The
allocation score is `predicted_relative_gain * predicted_error`.

Gain labels require reference-solver advances and validation ground truth, so
fitting the model is an offline training cost and is *not* charged to the test
budget. Only online features are used at deployment.

`oracle_gain` uses the true counterfactual gain on the test rollout as a
diagnostic upper bound for this family of allocators. It is deliberately
computed on the *uncorrected* rollout, so it upper-bounds a greedy one-step
objective rather than the true closed-loop optimum.

## 5. Controllers

| policy | score | budget-aware |
|---|---|---|
| `neural` | none | — |
| `numerical` | every step | cost 1.0 |
| `fixed` | static threshold on validation residual quantile | no |
| `adaptive` | online quantile with cumulative pacing | partially |
| `uniform` | evenly spaced steps | exact |
| `early` | first `k` steps | exact |
| `ranked_residual` | `residual_rel` | exact |
| `ranked_ensemble` | ensemble disagreement | exact |
| `ranked_calibrated` | step risk model | exact |
| `brace` | horizon risk model | exact |
| `brace_combo` | fusion of horizon risk and disagreement | exact |
| `gain_ranked` | predicted relative gain × predicted error | exact |
| `oracle_gain` | true counterfactual gain | exact |
| `oracle` | true one-step error | exact |
| `reanchor` | re-integrate from `u_0` to the chosen step | exact |

Budget-exact policies select the per-trajectory top-`k` scoring steps with
`k = floor(B*T)`; overlapping window corrections are excluded and windows that
would be truncated at the episode start are forbidden, so realised cost equals
the declared budget.

## 6. Metrics

- `rel_l2`: mean over trajectories of the relative `L2` error of the state,
  time-averaged over the horizon;
- `step_failure_rate`: fraction of (trajectory, step) pairs with relative error
  above `tau = 0.1`;
- `divergence_rate`: fraction of trajectories with mean relative error above 1;
- `cost_fraction`: realised solver intervals divided by `T`;
- seed-level 95% intervals use Student-t with `n-1` degrees of freedom;
- paired comparisons use a paired bootstrap over trajectories (5000 resamples),
  a paired t-test, Wilcoxon, and Cohen's `d_z`. With one seed the bootstrap
  interval is reported as a per-trajectory interval, not a seed interval.

## 7. Mechanism diagnostic

`scripts/analyze_gain_decay.py` measures, on held-out trajectories, the
counterfactual relative gain at every step and reports it by horizon decile,
together with the harm rate (fraction of steps where correction increases the
error) and the Spearman correlation between pre-correction error and relative
gain. This is the diagnostic that explains the allocation results: it does not
depend on any policy.
