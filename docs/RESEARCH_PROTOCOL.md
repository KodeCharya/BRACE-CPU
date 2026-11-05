# Research Protocol — BRACE-CPU

**Project:** BRACE-CPU — Budgeted Risk-Adaptive Computing Engine  
**Phase:** 2 — Research design  
**Date:** 2026-11-05  
**Status:** Draft for execution

---

## 1. Research questions

RQ1: Can lightweight risk features predict when a neural PDE surrogate will exceed an unacceptable prediction-error threshold?
RQ2: Does calibration improve prediction-risk reliability compared with an uncalibrated estimator?
RQ3: Can computational budgets be allocated more effectively using risk-ranked intervention than fixed-threshold correction?
RQ4: Does selective numerical correction improve long-horizon simulation reliability?
RQ5: What is the relationship between solver-intervention frequency, accuracy, latency, and computational cost?
RQ6: How robust is the method when physical parameters and initial-condition distributions change?

## 2. Central hypothesis

H0 (null): A lightweight neural surrogate + calibrated risk + budget-constrained numerical correction does NOT achieve a better accuracy-versus-cost trade-off than fixed-threshold correction, uncalibrated risk-based correction, or unconditional numerical computation.

H1 (alternative): It DOES achieve a better accuracy-versus-cost trade-off under the stated CPU/RAM constraints.

This is a hypothesis to investigate, not an established conclusion. Negative results will be reported honestly.

## 3. Mathematical problem definition

### Primary benchmark — 1D viscous Burgers' equation

u_t + u u_x = nu u_xx

Domain: x in [0, L], t in [0, T]
Boundary conditions: periodic (u(0,t) = u(L,t))
Initial condition: u(x,0) = u0(x), drawn from a parameterized random field
Parameter: nu (viscosity) varied over a range

### Secondary benchmark — 1D linear advection-diffusion

u_t + a u_x = D u_xx

Domain: x in [0, L], t in [0, T]
Boundary conditions: periodic
Initial condition: u(x,0) = u0(x)
Parameters: a (advection velocity), D (diffusion coefficient)

### Discretization

- Spatial grid: Nx = 64 (primary), Nx = 128 (optional comparison), uniform
- Time integrator: explicit/implicit finite difference; to be validated (Phase 3)
- Time step: dt chosen for stability (CFL/diffusion number)
- Reference solver: float64 finite difference, sufficiently accurate that discretization error is below the neural surrogate error target
- Grid/time-step convergence experiments conducted in Phase 3 to separate discretization error from neural prediction error

## 4. Numerical reference solver (Phase 3)

- Finite-difference discretization in space, finite-difference/RK time integration in time
- Periodic boundary handling
- Stability conditions verified (CFL, diffusion number)
- Convergence tests: compare solutions at multiple (Nx, dt) resolutions; confirm discretization error << surrogate error target
- Automated tests: solver correctness, boundary conditions, convergence
- float64 for reference; float32 for neural models

## 5. Dataset generation (Phase 4)

- Trajectories: ~300–1,000 initially (Burgers primary; advection-diffusion secondary)
- Spatial points: 64 (primary), 128 (comparison)
- Partitions (no trajectory leakage): train / validation / calibration / test
- Independent random seeds per partition
- Vary initial conditions and physical parameters (nu for Burgers; a, D for advection-diffusion)
- Out-of-distribution test set: parameter values and initial-condition distributions not in training
- Metadata recorded per trajectory: random seed, PDE parameters, grid resolution, time-step, solver, BCs, IC generation, trajectory ID, partition
- SHA-256 checksums saved for data files
- Stored under data/generation, data/processed, data/metadata

## 6. Lightweight neural surrogate (Phase 5)

- Primary model: small MLP, 2–4 hidden layers, 64–128 hidden units, ~10K–100K params
- Activation: ReLU or GELU (justified)
- Precision: FP32
- Framework: CPU-only PyTorch
- Formulation: predict next state u(t+dt) from current state u(t), physical parameters, time
- Autoregressive rollout for long-horizon; stability investigated
- Secondary: small DeepONet only if MLP stable (not mandatory)
- No large FNO as mandatory dependency

## 7. Risk estimation (Phase 6)

Prediction failure defined a priori: relative L2 error of next-step prediction > tolerance tau.

Candidate risk signals (evaluated independently, then combined):
1. Normalized PDE residual
2. Temporal prediction inconsistency
3. Physical constraint deviation
4. Prediction-history statistics
5. Ensemble disagreement (if affordable under RAM)

Risk score r in [0,1] estimates P(error > tau | features). Risk estimator must provide information beyond a simple residual threshold (tested, not assumed).

## 8. Calibration (Phase 6)

- Separate calibration dataset (never test labels)
- Methods: split-conformal-style error calibration, empirical risk calibration
- Assessed: risk-bin reliability, calibration error, failure-detection precision/recall
- Conformal = prediction-interval coverage distinguished from failure-probability calibration
- Distribution-free guarantees NOT claimed under arbitrary shift; exchangeability assumption reported
- Never use test-set labels to calibrate

## 9. Budget-aware controller (Phase 7)

Policies compared:
A. Always neural surrogate
B. Always numerical solver
C. Fixed residual threshold
D. Adaptive residual threshold
E. Uncalibrated risk-based intervention
F. Calibrated risk-based intervention
G. Budget-constrained risk-ranked intervention (BRACE-CPU)
H. Oracle intervention (uses actual future errors; evaluation only, never available to operational controller)

Budgets (intervention count and measured wall-clock cost): 5%, 10%, 20%, 40%, 60%
Budget defined as fraction of rollout steps allowed to invoke the numerical solver, and/or fraction of total compute time.
Every solver invocation measured for real wall-clock cost; never counted as free.
Correction: full-domain numerical advancement from current state (primary); one-step/multi-step/full fallback compared.

## 10. Evaluation metrics (Phase 8–11)

Accuracy: relative L2, MSE, max absolute error, long-horizon accumulated error.
Risk: precision, recall, F1, AUROC, risk-coverage curves, calibration error, false-negative intervention rate.
Performance: wall-clock latency, training time, inference time, correction time, invocation frequency, total CPU time, peak RAM, parameter count.
Reliability: long-horizon stability, error accumulation, failure detection, recovery after intervention, OOD performance.
CPU time distinguished from accuracy throughout.

## 11. Baselines (Phase 16)

1. Pure numerical solver
2. Pure neural surrogate
3. Fixed-threshold residual controller
4. Adaptive residual controller
5. Uncalibrated risk controller
6. Calibrated risk controller
7. Budget-constrained BRACE controller
8. Oracle reference

Consistent datasets, test trajectories, computational accounting across all.

## 12. Ablations (Phase 18)

Residual-only / temporal-consistency-only / physical-constraint-only / combined risk; with/without calibration; fixed vs adaptive budgets; calibration-set sizes; model sizes; spatial resolutions; rollout horizons.

## 13. Statistical methodology

- Multiple independent seeds: 3 for pilot, 5+ for main where feasible
- Report mean, std, confidence intervals, paired comparisons, effect sizes
- Identical test trajectories for paired comparisons
- Never report only best seed; never hide negative results

## 14. CPU performance engineering (Phase 20)

- Tune PyTorch and NumPy thread counts
- Small batches; small memory footprint
- Avoid unnecessary tensor copies
- Sequential execution; no multiprocessing unless profiling shows benefit within RAM
- Resource monitoring throughout; CPU resource report produced

## 15. Experiment management (Phase 22)

- Unique experiment ID, config file, seed, dataset checksum, model config, timestamp, runtime, peak memory, raw outputs, final metrics
- Incremental result saving; checkpoint recovery; raw results immutable; no silent overwrites

## 16. Deliverables

docs/ENVIRONMENT_REPORT.md (done), docs/LITERATURE_REVIEW.md (done), docs/NOVELTY_ANALYSIS.md (done), docs/RELATED_WORK_MATRIX.csv (done), docs/RESEARCH_PROTOCOL.md (this file), docs/PROJECT_STATUS.md, docs/EXPERIMENT_LOG.md, docs/DECISIONS.md, docs/RESOURCE_USAGE.md, docs/RESEARCH_FINDINGS.md, docs/REPRODUCIBILITY.md, docs/LIMITATIONS.md, paper/main.tex, paper/references.bib, paper/figures/, paper/tables/, paper/supplementary_material.md, source under src/, experiments under experiments/, tests under tests/, results under results/.

## 17. Success criteria (completion)

All phases complete; findings traceable to raw experimental evidence; honest publication-readiness assessment; no manuscript submission without explicit authorization.

---

*End of research protocol.*
