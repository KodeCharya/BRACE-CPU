# Literature Review — BRACE-CPU

**Project:** BRACE-CPU — Budgeted Risk-Adaptive Computing Engine  
**Date:** 2026-11-05  
**Scope:** Publications 2017–2026 relevant to neural PDE surrogates, uncertainty estimation, selective prediction, conformal calibration, and hybrid neural–numerical methods.  
**Sources consulted:** arXiv (verified directly via abstract pages). Web search was unavailable in this environment; all citations below were verified by fetching the arXiv abstract page. No citations are fabricated.

---

## 1. Neural PDE surrogates and neural operators

### 1.1 Fourier Neural Operator (FNO)
- **Title:** Fourier Neural Operator for Parametric Partial Differential Equations
- **Authors:** Zongyi Li, Nikola Kovachki, Kamyar Azizzadenesheli, Burigede Liu, Kaushik Bhattacharya, Andrew Stuart, Anima Anandkumar
- **Year:** 2020 (submitted 18 Oct 2020); published ICLR 2021
- **arXiv:** 2010.08895 — https://doi.org/10.48550/arXiv.2010.08895
- **Publication status:** arXiv preprint; ICLR 2021
- **Methodology:** Neural operator that learns the mapping from PDE parameters/initial conditions to the solution by parameterizing the integral kernel in Fourier space. Learns an entire family of PDEs rather than a single instance. Demonstrated on Burgers' equation, Darcy flow, Navier-Stokes. Claims up to three orders of magnitude speedup vs. traditional solvers and zero-shot super-resolution.
- **Evaluation benchmarks:** Burgers' equation, Darcy flow, Navier-Stokes equation.
- **Computational requirements:** GPU-based training/inference typical; spectral layers require FFTs.
- **Limitations:** Large spectral layers; memory-heavy at high resolution; not designed for selective/corrective computation; no uncertainty or risk calibration; no computational-budget gating.
- **Relationship to BRACE-CPU:** FNO is the canonical neural-operator baseline. BRACE-CPU deliberately uses a *lightweight MLP* rather than FNO because (a) the target hardware is CPU-only with 8 GiB RAM, and (b) the research question concerns risk-adaptive *selective* correction, which FNO does not address. FNO is cited as the high-end surrogate that BRACE-CPU's lightweight design contrasts with.

### 1.2 DeepONet
- **Title:** DeepONet: Learning nonlinear operators for identifying differential equations based on the universal approximation theorem of operators
- **Authors:** Lu Lu, Pengzhan Jin, George Em Karniadakis
- **Year:** 2019 (submitted 8 Oct 2019; revised 2020); published Nature Machine Intelligence 2021
- **arXiv:** 1910.03193 — https://doi.org/10.48550/arXiv.1910.03193 ; https://doi.org/10.1038/s42256-021-00302-5
- **Publication status:** arXiv preprint; Nature Machine Intelligence (2021)
- **Methodology:** Deep operator network with a branch net (encodes input function at sensors) and a trunk net (encodes output locations). Learns nonlinear operators between function spaces. Demonstrated on dynamic systems and PDEs. Claims high-order error convergence.
- **Evaluation benchmarks:** Dynamic systems and PDE operator identification.
- **Computational requirements:** Moderate; two-subnetwork architecture.
- **Limitations:** Requires sensor placement; operator-learning focus, not time-stepping surrogate; no uncertainty/risk estimation; no selective numerical correction.
- **Relationship to BRACE-CPU:** DeepONet is the second canonical neural-operator family. BRACE-CPU may evaluate a small DeepONet as a secondary model (per the project plan), but the primary model is an MLP rollout surrogate. Cited as related operator-learning work.

### 1.3 Physics-Informed Neural Networks (PINNs)
- **Title:** Physics Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear Partial Differential Equations
- **Authors:** Maziar Raissi, Paris Perdikaris, George Em Karniadakis
- **Year:** 2017 (submitted 28 Nov 2017); published JCP 2019
- **arXiv:** 1711.10561 — https://doi.org/10.48550/arXiv.1711.10561
- **Publication status:** arXiv preprint; Journal of Computational Physics (2019)
- **Methodology:** Neural networks trained to satisfy supervised data residuals *and* PDE residuals (physics-informed loss), yielding differentiable surrogates w.r.t. coordinates and parameters. Continuous-time and discrete-time formulations.
- **Evaluation benchmarks:** Nonlinear PDEs (Burgers, Navier-Stokes, etc.).
- **Computational requirements:** Training is expensive due to PDE-residual collocation; inference cheap.
- **Limitations:** Training cost; no uncertainty calibration; no selective correction; not focused on computational-budget trading.
- **Relationship to BRACE-CPU:** PINNs motivate the use of PDE-residual information as a risk signal (BRACE-CPU candidate risk feature #1: normalized PDE residual). BRACE-CPU differs by using the residual for *risk estimation and selective correction* at inference time rather than as a training loss.

---

## 2. Selective prediction and uncertainty calibration

### 2.1 Selective classification (reject option)
- **Title:** Selective Classification for Deep Neural Networks
- **Authors:** Yonatan Geifman, Ran El-Yaniv
- **Year:** 2017 (submitted 23 May 2017)
- **arXiv:** 1705.08500 — https://doi.org/10.48550/arXiv.1705.08500
- **Publication status:** arXiv preprint
- **Methodology:** Constructs a selective classifier given a trained DNN; user sets a desired risk level; test-time instances are rejected to meet the risk with high probability. Trade-off between coverage and risk. Demonstrated on CIFAR and ImageNet.
- **Evaluation benchmarks:** CIFAR, ImageNet (image classification).
- **Computational requirements:** Lightweight post-hoc wrapper on a trained classifier.
- **Limitations:** Classification setting (not regression/PDE); risk defined on classification error, not continuous PDE error; no numerical correction; no computational budget.
- **Relationship to BRACE-CPU:** Direct conceptual ancestor of BRACE-CPU's *selective intervention* idea: reject/delegate to a more expensive computation when risk is high. BRACE-CPU adapts this to *regression/rollout* PDE prediction with a numerical solver as the "referee" and adds a *computational budget* constraint absent from this work.

### 2.2 Conformalized Quantile Regression (CQR)
- **Title:** Conformalized Quantile Regression
- **Authors:** Yaniv Romano, Evan Patterson, Emmanuel J. Candès
- **Year:** 2019 (submitted 8 May 2019); NeurIPS 2019
- **arXiv:** 1905.03222 — https://doi.org/10.48550/arXiv.1905.03222
- **Publication status:** arXiv preprint; NeurIPS 2019
- **Methodology:** Combines conformal prediction (distribution-free finite-sample coverage) with quantile regression to produce adaptive, input-dependent prediction intervals with valid coverage without distributional assumptions.
- **Evaluation benchmarks:** Standard regression datasets.
- **Computational requirements:** Lightweight post-hoc calibration on a held-out calibration set.
- **Limitations:** Coverage guarantee is conditional on exchangeability (i.i.d. calibration/test); does not hold under arbitrary distribution shift; constructs intervals, not a scalar failure probability directly (though adaptable).
- **Relationship to BRACE-CPU:** CQR is the calibration-method reference point. BRACE-CPU investigates split-conformal-style calibration of a *failure probability* (error-exceeds-tolerance) on a separate calibration set. BRACE-CPU explicitly reports the exchangeability assumption and does not claim guarantees under distribution shift (see Phase 13 limitations). CQR is cited for the conformal calibration paradigm; BRACE-CPU's contribution is applying calibrated risk to *budget-constrained selective numerical correction* for PDE rollouts.

---

## 3. Topics surveyed but not yet tied to a single verified paper

The following topics are central to BRACE-CPU. I searched arXiv but did not fetch/verify a specific representative paper for each during this audit, so they are discussed conceptually without fabricated citations:

- **Residual-based error estimation for PDE surrogates:** Using the normalized PDE residual as an error/risk signal. Related conceptually to PINN residuals (Raissi et al., above) but applied at inference time for selective correction.
- **Adaptive numerical solvers / time-stepping:** Classical adaptive mesh refinement and adaptive time-stepping (standard numerical analysis). BRACE-CPU's controller is analogous in spirit (decide where/when to spend compute) but driven by a learned risk model rather than a local truncation-error estimator.
- **Long-horizon PDE rollout stability / error accumulation:** Known issue in autoregressive neural surrogates; error compounds over rollout steps. BRACE-CPU addresses via periodic numerical correction.
- **Computational budget allocation / risk-sensitive computation:** Allocating a limited compute budget across rollout steps based on risk ranking. No specific verified paper fetched.
- **Hybrid neural–numerical correction:** Combining a neural predictor with periodic numerical re-initialization/correction from the current state. Conceptually related to selective classification (delegate to referee) but not verified as a published PDE-specific method.

## 4. Summary of novelty position

| Element | Covered by existing work? | BRACE-CPU's angle |
|---|---|---|
| Neural PDE surrogate (MLP rollout) | Yes (many) | Lightweight CPU MLP, not FNO/DeepONet |
| Risk estimation for surrogate failure | Partially (residual used in PINN training) | Inference-time risk features + failure-probability model |
| Calibration of risk | Yes (conformal prediction / CQR) | Applied to PDE failure probability; explicit exchangeability assumption reported |
| Selective correction (delegate to solver) | Partially (selective classification, image) | Numerical solver as referee for PDE regression rollout |
| Computational budget on interventions | Not verified as a published PDE method | Budget-constrained *risk-ranked* intervention (novel combination) |
| CPU-only, 8 GiB constraint | Not a focus of prior work | Designed for and evaluated under tight CPU/RAM constraints |

**Novelty claim (tentative):** The specific combination of (a) a lightweight neural PDE surrogate, (b) calibrated failure-risk estimation, (c) budget-constrained risk-ranked selective numerical correction, and (d) evaluation under strict CPU/RAM constraints, is not represented by the verified papers above. BRACE-CPU's narrower contribution, if prior hybrid neural–numerical correction work exists, is the *calibration + budget-ranking* layer applied to lightweight CPU surrogates.

This will be revisited in docs/NOVELTY_ANALYSIS.md after a broader search.

---

*End of literature review. All arXiv IDs, titles, authors, and DOIs above were verified by fetching the arXiv abstract page.*
