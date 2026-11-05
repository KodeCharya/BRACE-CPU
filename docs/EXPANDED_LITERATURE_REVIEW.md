# Expanded Literature Review — BRACE-CPU

**Project:** BRACE-CPU — Budgeted Risk-Adaptive Computing Engine  
**Date:** 2026-11-05  
**Scope:** Focused review of neural PDE surrogates, hybrid neural–numerical solvers, adaptive/selective correction, residual-based error estimation, selective prediction, risk-aware inference, budget-constrained computation, conformal prediction, and OOD robustness.  
**Verification:** Every entry below was verified by fetching the arXiv abstract page. No citations fabricated.  
**Honest assessment:** BRACE-CPU's core concept (neural surrogate + residual-based error signal + selective numerical correction) is NOT novel — ANCHOR (Roy et al., arXiv:2512.19643) and several hybrid solvers (INC, HIN-LRI, MC², Perturbative-NeuSA, Error-Conditioned Neural Solvers) predate or overlap with it. BRACE-CPU's narrower, defensible contribution is the calibrated failure-probability + budget-constrained risk-ranked intervention package applied to lightweight CPU MLPs, with honest reporting that calibration adds no benefit.

---

## 1. Neural PDE surrogates and neural operators

### 1.1 Fourier Neural Operator (FNO)
- **Title:** Fourier Neural Operator for Parametric Partial Differential Equations
- **Authors:** Zongyi Li, Nikola Kovachki, Kamyar Azizzadenesheli, Burigede Liu, Kaushik Bhattacharya, Andrew Stuart, Anima Anandkumar
- **Year:** 2020 (submitted 18 Oct 2020); ICLR 2021
- **arXiv:** 2010.08895 — https://doi.org/10.48550/arXiv.2010.08895
- **Status:** arXiv preprint; ICLR 2021
- **Methodology:** Neural operator parameterizing the integral kernel in Fourier space; learns the mapping from PDE parameters/initial conditions to the solution across a family of PDEs.
- **Benchmarks:** Burgers' equation, Darcy flow, Navier-Stokes.
- **Computational requirements:** GPU-typical; FFT-based spectral layers; memory-heavy at high resolution.
- **Limitations:** Large spectral layers; no uncertainty/risk estimation; no selective correction; no computational budget.
- **Relationship to BRACE-CPU:** High-end surrogate baseline BRACE-CPU deliberately contrasts with (uses a lightweight MLP instead of FNO, suited to CPU-only 8 GiB RAM).

### 1.2 DeepONet
- **Title:** DeepONet: Learning nonlinear operators for identifying differential equations based on the universal approximation theorem of operators
- **Authors:** Lu Lu, Pengzhan Jin, George Em Karniadakis
- **Year:** 2019/2020; Nature Machine Intelligence 2021
- **arXiv:** 1910.03193 — https://doi.org/10.48550/arXiv.1910.03193 ; https://doi.org/10.1038/s42256-021-00302-5
- **Status:** arXiv preprint; Nature Machine Intelligence (2021)
- **Methodology:** Branch net (encodes input function at sensors) + trunk net (encodes output locations); learns nonlinear operators between function spaces.
- **Benchmarks:** Dynamic systems, PDE operator identification.
- **Limitations:** Sensor placement; operator-learning focus; no uncertainty; no selective correction; no budget.
- **Relationship to BRACE-CPU:** Secondary model option considered but not mandated; operator-learning context.

### 1.3 Physics-Informed Neural Networks (PINNs)
- **Title:** Physics Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear Partial Differential Equations
- **Authors:** Maziar Raissi, Paris Perdikaris, George Em Karniadakis
- **Year:** 2017; JCP 2019
- **arXiv:** 1711.10561 — https://doi.org/10.48550/arXiv.1711.10561
- **Status:** arXiv preprint; Journal of Computational Physics (2019)
- **Methodology:** PDE residual as a training loss; differentiable surrogate w.r.t. coordinates/parameters.
- **Limitation:** Residual used in training; no inference-time risk calibration; no selective correction; no budget.
- **Relationship to BRACE-CPU:** BRACE-CPU repurposes residual-like signals as inference-time risk features for selective correction (not as a training loss).

---

## 2. Hybrid neural–numerical solvers and adaptive/selective correction (DIRECTLY RELATED)

### 2.1 ANCHOR — Adaptive Numerical Correction for High-fidelity Operator Rollouts (DIRECT PREDECESSOR)
- **Title:** ANCHOR: Error-Controlled Adaptive Numerical Correction for Neural Operator Time Marching
- **Authors:** Rajyasri Roy, Dibyajyoti Nayak, Somdatta Goswami
- **Year:** 2025 (v1 22 Dec 2025); submitted 14 Jun 2026
- **arXiv:** 2512.19643 — https://doi.org/10.48550/arXiv.2512.19643
- **Status:** arXiv preprint
- **Methodology:** Online hybrid inference: pretrained neural operator as primary engine, adaptively coupled with a classical numerical solver using a physics-informed residual-based error estimator. Monitors an exponential moving average (EMA) of the normalized PDE residual to detect accumulating error and trigger corrective solver interventions, without ground truth. Evaluated on 1D/2D Burgers, 2D Allen-Cahn, 2D Cahn-Hilliard, 2D Navier-Stokes, 3D heat conduction.
- **Computational requirements:** Neural operator (not lightweight MLP); GPU-oriented.
- **Limitations:** Uses EMA of residual (not calibrated failure probability); no explicit computational budget or risk-ranked intervention; no wall-clock cost accounting; neural operator not lightweight MLP.
- **Relationship to BRACE-CPU:** **Direct conceptual predecessor.** ANCHOR demonstrates that residual-based selective numerical correction bounds long-horizon error. BRACE-CPU's narrower contribution: (i) lightweight MLP instead of neural operator, (ii) calibrated failure-probability estimation (not raw residual EMA), (iii) explicit budget-constrained risk-ranked (top-k) intervention with measured wall-clock accounting, (iv) CPU-only 8 GiB constraint. Honest finding: BRACE-CPU's calibration adds no benefit over raw risk, and threshold policies overshoot budgets.

### 2.2 INC — Indirect Neural Corrector for Auto-Regressive Hybrid PDE Solvers
- **Title:** INC: An Indirect Neural Corrector for Auto-Regressive Hybrid PDE Solvers
- **Authors:** Hao Wei, Aleksandra Franz, Bjoern List, Nils Thuerey
- **Year:** 2025; NeurIPS 2025
- **arXiv:** 2511.12764 — https://doi.org/10.48550/arXiv.2511.12764
- **Status:** arXiv preprint; NeurIPS 2025
- **Methodology:** Integrates learned corrections into the governing equations (indirect correction) rather than direct state updates, reducing error amplification during long rollouts.
- **Limitation:** Not selective/budgeted; no uncertainty calibration; no wall-clock accounting.
- **Relationship to BRACE-CPU:** Related hybrid corrector; BRACE-CPU differs by being selective/budgeted with risk ranking.

### 2.3 HIN-LRI — Hybrid Iterative Neural Low-Regularity Integrator
- **Title:** Hybrid Iterative Neural Low-Regularity Integrator for Nonlinear Dispersive Equations
- **Authors:** Zhangyong Liang, Huanhuan Gao
- **Year:** 2026 (v1 6 May 2026)
- **arXiv:** 2605.04853 — https://doi.org/10.48550/arXiv.2605.04853
- **Status:** arXiv preprint
- **Methodology:** Classical low-regularity integrator + lightweight neural operator correcting the residual defect on a low-dimensional latent manifold; solver-in-the-loop training.
- **Limitation:** Hybrid corrector, not selective/budgeted; no uncertainty calibration.
- **Relationship to BRACE-CPU:** Related hybrid correction; BRACE-CPU adds selective risk-ranked budget enforcement.

### 2.4 MC² — Monte Carlo Correction for Fast Elliptic PDE Solving
- **Title:** MC²: Monte Carlo Correction for Fast Elliptic PDE Solving
- **Authors:** Ethan Hsu, Hong Meng Yam, Ivan Ge
- **Year:** 2026 (submitted 9 May 2026)
- **arXiv:** 2605.09288 — https://doi.org/10.48550/arXiv.2605.09288
- **Status:** arXiv preprint
- **Methodology:** Hybrid Walk-on-Spheres (Monte Carlo) + neural correction; treats low-budget MC as structured estimator, learns single-pass neural correction.
- **Limitation:** Elliptic PDEs only; not time-marching/selective; no budget enforcement over rollout.
- **Relationship to BRACE-CPU:** Related hybrid correction concept (fast neural + structured numerical estimator).

### 2.5 Error-Conditioned Neural Solvers (ENS)
- **Title:** Error-Conditioned Neural Solvers
- **Authors:** Haina Jiang, Liam Wang, Peng-Chen Chen, Min Seop Kwak, Seungryong Kim, Brian Bell, Jeong Joon Park
- **Year:** 2026 (v1 25 Jun 2026)
- **arXiv:** 2606.27354 — https://doi.org/10.48550/arXiv.2606.27354
- **Status:** arXiv preprint
- **Methodology:** Passes the PDE residual field as a direct input to the network at each iteration, enabling it to read its own error structure and learn an iterative correction policy.
- **Limitation:** Not selective/budgeted; no uncertainty calibration; no wall-clock accounting.
- **Relationship to BRACE-CPU:** Related use of residual as error signal; BRACE-CPU uses residual as one of several risk features for selective intervention.

### 2.6 Perturbative-NeuSA
- **Title:** Perturbative-NeuSA: A Structured Spectral Framework for Time-Dependent PDEs
- **Authors:** Xianli Zhu, Jia Yin
- **Year:** 2026 (submitted 27 Jul 2026)
- **arXiv:** 2607.24345 — https://doi.org/10.48550/arXiv.2607.24345
- **Status:** arXiv preprint
- **Methodology:** Decomposes solution into low-fidelity background + high-resolution perturbation; only unresolved dynamics learned; neural closure conditional on background fidelity.
- **Limitation:** Not selective/budgeted; no uncertainty calibration.
- **Relationship to BRACE-CPU:** Related residual/correction framing; BRACE-CPU's selective intervention is distinct.

### 2.7 Multilevel Neural Corrections with Adaptive Sampling
- **Title:** Multilevel Neural Corrections with Adaptive Sampling for Partial Differential Equations
- **Authors:** Yu Yang, Qiaolin He
- **Year:** 2026 (v1 17 Jan 2026)
- **arXiv:** 2601.12000 — https://doi.org/10.48550/arXiv.2601.12000
- **Status:** arXiv preprint
- **Methodology:** Multi-level framework; adaptive sampling from normalized residual-gradient indicators; successive neural corrections across levels.
- **Limitation:** Not selective/budgeted rollout; no uncertainty calibration; no wall-clock accounting.
- **Relationship to BRACE-CPU:** Related residual-based error indicator; BRACE-CPU uses residual as a risk feature for budgeted intervention.

### 2.8 Transferability of Learned States in Neural PDE Solvers
- **Title:** Transferability of Learned States in Neural PDE Solvers
- **Authors:** Shunye Wang, Haochen Wen, Shuo Li Liu, Xuanyi Wang, Lihao Liu, Zhongying Deng
- **Year:** 2026 (v1 7 Oct 2026)
- **arXiv:** 2610.10972 — https://doi.org/10.48550/arXiv.2610.10972
- **Status:** arXiv preprint
- **Methodology:** Reuse contract separating solution accuracy, learning contribution, and numerical utility; paired state comparisons; cost accounting; matched target information and budgets.
- **Limitation:** Not selective correction over rollout; no uncertainty calibration.
- **Relationship to BRACE-CPU:** Related emphasis on cost accounting and budget-aware neural/numerical reuse.

### 2.9 SNAP — Safeguarded Neurally-Accelerated Preconditioning
- **Title:** Safeguarded Neurally-Accelerated Preconditioning (SNAP) framework for Nonlinear Optimization
- **Authors:** Youngkyu Lee, George Em Karniadakis, Alena Kopaničáková
- **Year:** 2026 (submitted 28 Oct 2026)
- **arXiv:** 2610.38790 — https://doi.org/10.48550/arXiv.2610.38790
- **Status:** arXiv preprint
- **Methodology:** Hybridizes nonlinear optimization with pre-trained neural operators; a posteriori safeguard accepts only corrections satisfying descent/acceptance conditions; proves global convergence.
- **Limitation:** Optimization/preconditioning context, not selective rollout correction; no budget enforcement over rollout.
- **Relationship to BRACE-CPU:** Related "safeguard" concept (accept only corrections meeting criteria); BRACE-CPU's safeguard is risk-ranked + budgeted.

---

## 3. Selective prediction, risk-aware inference, and budget-constrained computation

### 3.1 Selective Classification for Deep Neural Networks
- **Title:** Selective Classification for Deep Neural Networks
- **Authors:** Yonatan Geifman, Ran El-Yaniv
- **Year:** 2017
- **arXiv:** 1705.08500 — https://doi.org/10.48550/arXiv.1705.08500
- **Status:** arXiv preprint
- **Methodology:** Reject-option selective classifier; user sets desired risk level; test-time instances rejected to meet risk with high probability; coverage–risk trade-off.
- **Limitation:** Image classification (not regression/PDE); no numerical solver referee; no computational budget.
- **Relationship to BRACE-CPU:** Conceptual ancestor of selective intervention; BRACE-CPU adapts to PDE regression with numerical solver referee + budget.

### 3.2 A trust layer for fast pedestal-stability surrogates
- **Title:** A trust layer for fast pedestal-stability surrogates: calibrated error bars, out-of-distribution alarms and few-shot recalibration
- **Authors:** Amin Vakhshouri
- **Year:** 2026 (submitted 4 Nov 2026)
- **arXiv:** 2611.06597 — https://doi.org/10.48550/arXiv.2611.06597
- **Status:** arXiv preprint
- **Methodology:** Model-agnostic trust layer: conformal error bars, OOD alarms routing doubtful cases to the physics code, few-shot recalibration. Tests on fusion stability surrogates; error bars hold nominal coverage in-distribution, fall short under plasma-shape shift; routing flagged cases to physics code restores coverage.
- **Limitation:** Fusion stability surrogates (not time-marching PDE rollout); no computational budget enforcement.
- **Relationship to BRACE-CPU:** Closely related — routes uncertain cases to a physics/numerical code; BRACE-CPU differs by budget-constrained risk-ranked intervention + wall-clock accounting + CPU-only focus.

### 3.3 Conformalized Quantile Regression (CQR)
- **Title:** Conformalized Quantile Regression
- **Authors:** Yaniv Romano, Evan Patterson, Emmanuel J. Candès
- **Year:** 2019; NeurIPS 2019
- **arXiv:** 1905.03222 — https://doi.org/10.48550/arXiv.1905.03222
- **Status:** arXiv preprint; NeurIPS 2019
- **Methodology:** Conformal prediction + quantile regression; distribution-free finite-sample coverage intervals without distributional assumptions.
- **Limitation:** Coverage assumes exchangeability; intervals not failure probabilities; no budget; no numerical correction.
- **Relationship to BRACE-CPU:** Calibration-method reference point; BRACE-CPU calibrates a scalar failure probability (error > tolerance) for intervention decisions and explicitly reports the exchangeability assumption.

### 3.4 Conformal Prediction for Distribution-to-Distribution Regression
- **Title:** Conformal Prediction for Distribution-to-Distribution Regression
- **Authors:** Trung-Khang Tran, Tuan A. Hoang, Viet-Hoang Tran, Tan M. Nguyen
- **Year:** 2026 (submitted 4 Nov 2026)
- **arXiv:** 2611.06320 — https://doi.org/10.48550/arXiv.2611.06320
- **Status:** arXiv preprint
- **Methodology:** Conformal prediction for distribution-to-distribution regression; orthogonal basis space for interpretable prediction sets; adaptive variant with conditional coverage.
- **Limitation:** Not PDE/time-marching; no selective correction; no budget.
- **Relationship to BRACE-CPU:** Related conformal prediction extension; BRACE-CPU uses Platt scaling (simpler) and reports null calibration result.

---

## 4. Literature comparison matrix

| Prior work | Neural surrogate | Risk estimation | Selective correction | Explicit budget | Actual wall-clock accounting | Key distinction from BRACE-CPU |
|---|---|---|---|---|---|---|
| FNO (Li et al., 2020) | Yes (neural operator) | No | No | No | No | Large GPU operator; no risk/selective/budget |
| DeepONet (Lu et al., 2020) | Yes (operator net) | No | No | No | No | No risk/selective/budget |
| PINNs (Raissi et al., 2017) | Yes (physics loss) | No (training loss) | No | No | No | Residual in training, not inference risk |
| Geifman & El-Yaniv (2017) | Yes (classifier) | Yes (risk score) | Yes (reject) | No | No | Image classification; no PDE/numerical referee/budget |
| Romano et al. CQR (2019) | No | Yes (conformal intervals) | No | No | No | Interval coverage, not failure probability/budget |
| **ANCHOR (Roy et al., 2025)** | Yes (neural operator) | Yes (residual EMA) | Yes (numerical solver) | No | No | **Closest predecessor; no calibrated probability, no budget, no wall-clock, not lightweight MLP** |
| INC (Wei et al., 2025) | Yes (hybrid corrector) | No | Yes (indirect correction) | No | No | Not selective/budgeted; no calibration |
| HIN-LRI (Liang & Gao, 2026) | Yes (hybrid corrector) | No | Yes (residual defect) | No | No | Not selective/budgeted; no calibration |
| MC² (Hsu et al., 2026) | Yes (hybrid NN) | No | Yes (Monte Carlo + NN) | No | No | Elliptic only; not time-marching/budget |
| ENS (Jiang et al., 2026) | Yes (error-conditioned) | Yes (residual as input) | Yes (iterative correction) | No | No | Not selective/budgeted; no calibration |
| Perturbative-NeuSA (Zhu & Yin, 2026) | Yes (perturbation) | No | Yes (neural closure) | No | No | Conditional closure; not selective/budgeted |
| Multilevel Neural Corrections (Yang & He, 2026) | Yes (multi-level) | Yes (residual-gradient) | Yes (successive correction) | No | No | Not selective rollout/budget; no calibration |
| Trust layer (Vakhshouri, 2026) | Yes (surrogate) | Yes (conformal/OOD alarm) | Yes (route to physics code) | No | No | Fusion surrogate; no budget over rollout; no wall-clock |
| **BRACE-CPU (this work)** | Yes (lightweight MLP) | Yes (calibrated failure prob.) | Yes (numerical solver) | **Yes (intervention count + wall-clock)** | **Yes (per-invocation timing)** | **Calibrated + budget-constrained risk-ranked selective correction for lightweight CPU MLP; honest null calibration result** |

---

## 5. Novelty position (honest)

BRACE-CPU is **not** the first:
- neural PDE surrogate (FNO, DeepONet, PINNs predate it),
- hybrid neural–numerical solver (ANCHOR, INC, HIN-LRI, MC², ENS, Perturbative-NeuSA predate it),
- residual-based error signal (ANCHOR, ENS, Yang & He predate it),
- selective prediction (Geifman & El-Yaniv predates it),
- conformal/uncertainty calibration (Romano et al., Vakhshouri predate it),
- selective numerical correction with residual monitoring (**ANCHOR directly anticipates BRACE-CPU**).

**BRACE-CPU's narrower, defensible contribution** is the specific combination of:
1. A **lightweight MLP** (~17K parameters, FP32, CPU-only) rather than a neural operator,
2. **Calibrated failure-probability estimation** (Platt scaling on held-out calibration data) rather than raw residual monitoring,
3. **Budget-constrained, risk-ranked (top-k) selective numerical intervention** with explicit intervention-count **and** measured wall-clock budgets,
4. Evaluation under strict **CPU-only, 8 GiB RAM** constraints with real per-invocation cost accounting,
5. Honest reporting that **calibration adds no measurable benefit** (null result) and that **threshold policies overshoot their nominal budgets** under error accumulation.

If a reviewer considers the calibrated + budget-ranked combination insufficiently novel given ANCHOR + trust-layer work, the contribution should be narrowed to: *"An empirical CPU-focused investigation showing that (a) budget-constrained risk-ranked selective numerical correction enforces computational budgets exactly whereas threshold policies overshoot under error accumulation, and (b) calibrated failure-probability estimation provides no measurable benefit over raw risk scores for lightweight neural Burgers' surrogates."*

---

*End of expanded literature review.*
