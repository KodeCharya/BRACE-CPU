# Results

All values below are read from `results/tables/*.csv` and
`results/processed/results_digest.{json,txt}`. They are means over seeds
(Burgers test: 2 seeds; everything else: 1 seed) and over test trajectories.
Cost is the realised fraction of full-solver cost.

## 1. Correction works, and spending is exactly as declared

Burgers-1D, test split, surrogate-only error 0.0611:

| policy | B=0.05 | B=0.1 | B=0.2 | B=0.4 |
|---|---|---|---|---|
| uniform | 0.0492 | 0.0408 | 0.0314 | 0.0210 |
| early | 0.0587 | 0.0554 | 0.0422 | **0.0178** |
| ensemble-ranked | 0.0455 | **0.0387** | **0.0298** | 0.0188 |
| BRACE (horizon risk) | **0.0451** | 0.0397 | 0.0322 | 0.0215 |
| BRACE-fusion | 0.0452 | 0.0391 | 0.0312 | 0.0200 |
| gain-ranked | 0.0473 | 0.0429 | 0.0364 | 0.0269 |
| gain oracle | 0.0459 | 0.0406 | 0.0332 | 0.0242 |
| true-error oracle | 0.0509 | 0.0478 | 0.0426 | 0.0345 |

Every ranked policy realises its declared budget exactly (max deviation 0.000
over all 20 main runs per policy). The `fixed` threshold baseline is **not**
budget-aware: it overspends in 11 of 20 runs by up to +0.699, reaching cost 1.00
(and error 0) on shifted splits. `adaptive` underspends in 13 of 20 runs by up
to 0.222.

## 2. The learned risk model does not beat a training-free score

- At B ≤ 0.2 the training-free **ensemble-disagreement** ranking is the best or
  tied-best policy on all three in-distribution benchmarks.
- The calibrated horizon-risk model (`brace`) is 0.0451/0.0397/0.0322/0.0215 on
  Burgers test, i.e. within 1–4% of ensemble ranking but never clearly ahead;
  at B = 0.4 it is 14% worse than `early`.
- Fusing the two (`brace_combo`) does not help: the paired difference against
  ensemble ranking is +0.0005/+0.0014/+0.0012 on Burgers test, all with CIs
  spanning zero (one of two seeds significant at B ≥ 0.2, `d_z = 0.31/0.43`).
- `ranked_calibrated` (step risk model) and the **true-error oracle** are the
  *worst* ranked policies on advection–diffusion (0.488 and 0.488 at B = 0.4 vs
  0.207 for `early`).

## 3. A learned gain model is worse than the score it replaces

`gain_ranked` is worse than `ranked_ensemble` at every budget on all three
in-distribution benchmarks (Burgers test: +0.0018 to +0.0080; Allen–Cahn:
+0.0002 to +0.0014; advection–diffusion: +0.0021 to +0.0322), and on Burgers
OOD-parameter by +0.0140 to +0.0510 (all seeds significant, `d_z` up to 2.07).

The one apparent exception is instructive. On severely shifted
advection–diffusion, `gain_ranked` is *better* than ensemble ranking
(1.050 vs 1.076 at B = 0.1; 0.560 vs 0.650 at B = 0.4) — but only because its
predicted relative-gain scores become nearly constant under shift, so the
top-k tie-break degenerates into selecting the first *k* steps. Its error equals
the `early` schedule to four decimals at four separate budget/split points.
That is not a working gain model; it is a gain model whose output has collapsed
and is silently being read as "spend early".

Even the **counterfactual gain oracle loses to the trivial early schedule at
large budgets**:

| setting | early | gain oracle | true-error oracle |
|---|---|---|---|
| Burgers test, B=0.4 | 0.0178 | 0.0242 | 0.0345 |
| Allen–Cahn test, B=0.4 | 0.0029 | 0.0046 | 0.0062 |
| Advection–diffusion test, B=0.4 | 0.2071 | 0.2181 | 0.4879 |

So the greedy one-step objective itself — not merely our estimate of it — is the
wrong objective for budgeted correction.

## 4. Mechanism: correction value decays along the horizon

`results/tables/gain_decay.csv`, `results/figures/gain_decay.png`:

| benchmark / split | relative gain first → last decile | harm rate | Spearman(pre-error, relative gain) |
|---|---|---|---|
| Burgers test | 0.236 → 0.119 | 0.3% | **+0.008** |
| Burgers ood_param | 0.195 → 0.024 | 3.7% | **−0.843** |
| Allen–Cahn test | 0.155 → 0.015 | 0.0% | **−0.665** |
| Advection–diffusion test | 0.149 → 0.009 | 0.1% | **−0.842** |

Two consequences:

1. In-distribution, the surrogate's own error carries essentially no information
   about where a correction *helps* (rho ≈ 0 on Burgers). Risk ranking is
   therefore no better than ranking by ensemble spread.
2. Under shift, the relationship inverts: the highest-error steps are the steps
   where correcting helps least, because the state has left the manifold on
   which the reference integrator is informative. This is why the true-error
   oracle is the worst allocator under shift.

This is also why front-loading wins at large budgets: early corrections act on
surrogate states that are still close to the reference.

## 5. Out-of-distribution: coverage fails, compliance holds

Empirical conformal coverage at nominal 0.9:

| split | Burgers | Allen–Cahn | Advection–diffusion |
|---|---|---|---|
| test | 0.929 / 0.944 | 0.817 | 0.879 |
| ood_param | 0.430 / 0.490 | 0.550 | 0.045 |
| ood_ic | 0.585 | 0.209 | 0.007 |

Upper-bound coverage stays at 0.96–1.00, so the intervals do not fail in the
"too narrow" direction that would be dangerous for a safety claim, but the
point estimates are not exchangeable under shift and the nominal guarantee does
not transfer.

Budget compliance is unaffected by shift for ranked policies (exact by
construction) but the *benefit* shrinks: on advection–diffusion `ood_param` the
surrogate error is 1.195 and best budgeted correction at B = 0.4 is 0.625; on
`ood_ic` it is 1.289 → 0.560. Corrections mitigate, they do not recover.

## 6. Ablations: ensemble information is the only component that matters

Burgers test, seed 0, `brace` at B = 0.2 / 0.4:

| variant | B=0.2 | B=0.4 |
|---|---|---|
| full | 0.0324 | 0.0205 |
| no_temporal | 0.0300 | 0.0185 |
| no_ensemble_feat | 0.0320 | 0.0197 |
| probability head | 0.0304 | 0.0196 |
| myopic expected error | 0.0318 | 0.0198 |
| gamma = 0.99 | 0.0333 | 0.0182 |
| gamma = 0.90 | 0.0345 | 0.0241 |
| residual-only features | 0.0362 | 0.0266 |
| **single member (no ensemble)** | **0.0416** | **0.0267** |

Removing the ensemble costs +28%/+31%. Removing every other component is within
noise of the full model, and several removals *improve* it. The ablation
therefore says: the useful signal is disagreement between independently trained
surrogates, not the risk model built on top of it.

## 7. Correction granularity beats correction volume

Window corrections (integrate `W` intervals per action) at matched budget,
Burgers test:

| policy | step, B=0.4 | window W=5, B=0.4 |
|---|---|---|
| ensemble-ranked | 0.0188 | 0.0224 |
| BRACE-fusion | 0.0200 | 0.0221 |
| BRACE | 0.0215 | 0.0238 |
| residual-ranked | 0.0275 | 0.0292 |

Re-anchoring (re-integrate from `u_0` to the chosen step) is worse still
(0.0342). Fine-grained, well-placed corrections dominate coarser ones at equal
cost.

## 8. DeepONet is exploratory only

One member, 150 epochs, test split: mean 0.3714, median 0.2854, 1/40 trajectories
diverges (mean inflated by that trajectory; the same member diverges on 1/80
validation trajectories). Corrections lower the median to 0.2294 at B = 0.4 but
do not eliminate divergence. With one member there is no ensemble signal and no
seed-level uncertainty; these numbers are reported to show the correction engine
is architecture-agnostic, not as a DeepONet-versus-FNO comparison.

## 9. Known measurement caveat

At the 32×32 resolution used for advection–diffusion, one exact reference
interval costs 0.188 ms while one FNO-2D forward pass on the same grid costs
0.836 ms per sample. The surrogate is therefore **more expensive than the exact
solver** for that benchmark, so BRACE's speedup argument does not apply there;
only the error-reduction question is meaningful. For Burgers-1D the ratio is
favourable (0.133 ms surrogate vs 2.10 ms solver per interval) and for
Allen–Cahn it is 2.5×.

The DeepONet benchmark is also more expensive to plan for than to run: its
per-sample latency is 0.007 ms but it is far less accurate, so the trade-off is
not favourable on this problem.
