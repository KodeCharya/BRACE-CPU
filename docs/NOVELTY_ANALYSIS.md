# Novelty Analysis — BRACE-CPU

**Project:** BRACE-CPU — Budgeted Risk-Adaptive Computing Engine  
**Date:** 2026-11-05  
**Status:** Tentative — to be confirmed/revised after Phase 9 main experiments.

---

## 1. Research question (re-stated)

> Can a lightweight neural PDE surrogate achieve a better accuracy-versus-computational-cost trade-off by using calibrated prediction-risk estimates to selectively invoke numerical computation?

## 2. What already exists (verified literature)

| Capability | Verified prior work | Gap relative to BRACE-CPU |
|---|---|---|
| Neural operator for PDEs | FNO (Li et al., 2020/2021); DeepONet (Lu et al., 2020/2021) | These are large/spectral operator models, not lightweight CPU MLPs; no risk calibration; no selective correction |
| Physics-informed training loss | PINNs (Raissi et al., 2017/2019) | Residual used as a *training* loss; BRACE-CPU uses residual as an *inference-time risk signal* |
| Selective prediction / reject option | Geifman & El-Yaniv (2017) | Image classification; no numerical solver referee; no computational budget |
| Conformal uncertainty calibration | Romano et al. CQR (2019) | Prediction-interval coverage; BRACE-CPU calibrates a scalar failure probability (error > tolerance) for intervention decisions |

## 3. The specific combination not found in verified literature

The combination of **all four** of the following, evaluated under strict CPU/RAM constraints:

1. Lightweight neural PDE surrogate (MLP autoregressive rollout, ~10K–100K params, FP32, CPU)
2. Calibrated prediction-risk estimation (failure probability that error exceeds a tolerance, calibrated on a held-out set)
3. Budget-constrained, risk-ranked selective numerical correction (intervene only on highest-risk steps within a compute budget)
4. Numerical solver as referee, invoked from the current state (no future ground truth), with measured wall-clock cost accounting

is not represented in the papers verified during this literature review (Section 2). In particular:
- Selective classification (Geifman & El-Yaniv) delegates to a human/abstains, not to a numerical solver, and has no compute budget.
- Conformal methods (CQR) produce intervals but are not coupled to a budget-constrained numerical intervention policy.
- PINNs use the residual in training, not as a calibrated inference-time risk for selective correction.

## 4. Narrower defensible contribution (fallback)

If a published hybrid neural–numerical correction method is later found that overlaps, BRACE-CPU's narrower contribution remains defensible:

> **Calibrated, budget-constrained risk-ranked selective numerical correction for lightweight CPU neural PDE surrogates**, with explicit accounting of intervention wall-clock cost, failure-probability calibration (not just interval coverage), and evaluation under distribution shift and long-horizon rollout.

That is: the novelty is not "neural + numerical" generally, but the **risk-calibration + budget-ranking + CPU-constrained** package.

## 5. What BRACE-CPU does NOT claim

- Not the first neural PDE surrogate.
- Not the first neural operator.
- Not the first conformal/uncertainty-calibration method.
- Not the first selective-prediction method.
- Not distribution-free under arbitrary shift (exchangeability assumption reported explicitly; OOD tested empirically, no guarantee claimed).
- Not first to combine a neural model with a numerical solver *at all* (if such work exists, the calibration+budget layer is the narrower claim).

## 6. Risks to novelty

- A hybrid neural–numerical corrector for PDEs may exist in the computational-physics / scientific-ML literature (e.g., "learning to correct" or "neural–numerical hybrid solvers"). This literature review did not verify one. If found during Phase 9–11, the novelty statement will be narrowed accordingly (see Section 4).
- The use of PDE residuals as risk signals is implicit in PINN literature; BRACE-CPU's distinct use is inference-time, calibrated, budget-gated intervention.

## 7. Decision

Proceed with implementation. Treat the claim as "tentative novelty: calibrated budget-constrained risk-ranked selective numerical correction for lightweight CPU surrogates." Revisit and narrow in docs/NOVELTY_ANALYSIS.md after main experiments if overlapping prior work surfaces.

---

*End of novelty analysis.*
