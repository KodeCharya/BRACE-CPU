# Limitations — BRACE-CPU

**Project:** BRACE-CPU — Budgeted Risk-Adaptive Computing Engine  
**Date:** 2026-11-05 (updated after independent audit)

## 1. Scientific scope

- Primary benchmark: 1D viscous Burgers' equation only; secondary: 1D linear advection-diffusion. Results may not generalize to multi-dimensional, turbulent, or stiff PDEs.
- Single primary surrogate: a small MLP (~17K parameters, 3 hidden layers × 64 units, ReLU). Larger neural operators (FNO, DeepONet) were intentionally not made a mandatory dependency due to CPU/RAM constraints and were not exhaustively evaluated.
- Failure tolerance τ = 0.01 set a priori; results are sensitive to this choice.

## 2. Calibration assumptions

- Failure-probability calibration (Platt scaling) assumes exchangeability between the calibration set and the test set. **No distribution-free guarantees are claimed under arbitrary distribution shift.** Out-of-distribution performance is evaluated empirically but without a theoretical guarantee.
- Conformal prediction-interval coverage is distinct from failure-probability calibration; we report the latter and do not conflate the two.
- Calibration provided **no measurable benefit** in this setting (ECE/Brier/AUROC identical to uncalibrated; paired p=0.29). This null result is reported honestly.

## 3. Resource constraints

- CPU-only, 8 GiB RAM, 4 cores. Model and dataset sizes are small by design. Wall-clock times and solver costs are hardware-specific and may not transfer to other machines.
- No multiprocessing; experiments run sequentially.
- BRACE's one-time pass-1 preparation cost (~54 ms/rollout) is reported separately and not amortized into per-step wall time.

## 4. Implementation scope

- Numerical correction is full-domain one-step advancement from the current state. Localized correction was considered but not implemented (not mathematically/numerically defended for this phase).
- The oracle policy uses ground truth for evaluation only and is never available to the operational controller.
- G's pass-1 risk pre-scoring runs on an all-surrogate trajectory; once interventions begin, the controlled state may diverge from the pre-scored trajectory — a known approximation, acknowledged as a limitation.

## 5. Statistical limitations

- 5 independent seeds for main experiments; confidence intervals are normal-approximation (mean ± 1.96 · SE).
- Small test set (25 trajectories per seed) limits precision of tail-risk estimates.
- **Training nondeterminism:** Retraining seed 0 from scratch (same seed, same data) produced moderately different models (relL2 differences up to ~0.04 on hard policies), indicating PyTorch CPU training nondeterminism. Results are reported as mean ± std/CI across seeds, which absorbs this variation. Bitwise reproducibility of retraining is not claimed.

## 6. Novelty boundaries

- BRACE-CPU is not the first neural PDE surrogate, the first uncertainty-calibration method, or the first selective-prediction method. Its tentative novelty is the calibrated + budget-constrained risk-ranked selective numerical correction package applied to lightweight CPU surrogates. This claim remains tentative pending broader literature confirmation.

## 7. OOD robustness

- Out-of-distribution (unseen low viscosity ν = 0.005–0.0079 vs train 0.01–0.03; different IC distribution): failure rate ~0.67 (vs ~0.09 in-distribution), AUROC degrades to ~0.90. BRACE becomes less reliable under genuine extrapolation.

## 8. Publication status

- Manuscript prepared locally; not submitted to any journal or conference. No claim of acceptance. Explicit authorization required before any submission.
