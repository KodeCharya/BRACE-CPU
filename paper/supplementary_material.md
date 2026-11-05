# Supplementary Material — BRACE-CPU

**Project:** BRACE-CPU — Budgeted Risk-Adaptive Computing Engine  
**Date:** 2026-11-05

## Contents

1. Numerical solver details (discretization, stability, convergence)
2. Risk feature definitions (full derivation)
3. Calibration method (Platt scaling) details
4. Controller policy pseudocode (A–H)
5. Cost accounting model (wall-clock, intervention count)
6. Full per-seed results tables
7. Experiment configs and seeds

## 1. Numerical solver

- Burgers: $u_t + (0.5 u^2)_x = \nu u_{xx}$, 2nd-order central differences, periodic BC, RK4 time integration.
- Stability: $dt \le \text{safety}\cdot\min(dx/\max|u|,\; dx^2/(2\nu))$, safety $=0.35$ (generation) / RK4 stable.
- Reference precision: float64. Surrogate precision: float32.
- Convergence: verified against manufactured exact solution of advection-diffusion; error $<10^{-2}$ rel L2 at $n_x=128$.

## 2. Risk features

Six features per step (no future ground truth):
1. Normalized PDE residual $\|du/dt - (-\nu u_{xx} + u u_x)\|/\|u\|$
2. Temporal inconsistency: one full step vs two half-steps
3. Mass-conservation deviation: $|\bar u_{pred} - \bar u_t|/|\bar u_t|$
4. Increment magnitude $\|u_{pred}-u_t\|/\|u_t\|$
5. Gradient magnitude $\|u_x\|/\|u\|$
6. Viscosity $\nu$

## 3. Calibration

- Logistic risk model on features $\to$ raw logit
- Platt scaling (1D logistic regression) on calibration-set logits $\to$ calibrated $P(\text{failure})$
- Metrics: ECE, Brier score, AUROC, precision/recall/F1 at threshold 0.5

## 4. Controller policies

A always_neural; B always_numerical; C fixed_threshold; D adaptive_threshold; E uncalibrated_risk; F calibrated_risk; G budget_risk_ranked (BRACE-CPU); H oracle (evaluation only).

G pre-scores all steps on an all-surrogate trajectory and intervenes on the top-$k=\text{budget}\cdot n\_steps$ riskiest steps.

## 5. Cost accounting

Every neural inference and every solver invocation timed with `time.perf_counter`. Total = neural + solver wall-clock. Intervention count = number of solver calls. Budget = max intervention fraction.

## 6. Full per-seed results

See `results/processed/summary_by_policy.csv` and `results/raw/<experiment_id>_all.json`.

## 7. Experiment config

See `results/raw/<experiment_id>/config.json` (seeds, budgets, tau, epochs, stride, dataset checksum).
