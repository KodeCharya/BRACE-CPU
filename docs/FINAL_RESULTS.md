# Final Results — BRACE-CPU

**Date:** 2026-11-05  
**Status:** Audit-passed, ready for manuscript finalization  
**Basis:** Corrected main experiment `exp_20261105_134910_63a5fa94` (5 seeds × 25 trajectories × 8 policies × budgets 5/10/20/40/60%). CPU-only (Intel i5-6500, 4 cores, 8 GiB RAM).

---

## Headline numbers (5 seeds, mean ± 95% CI)

### Accuracy vs pure neural (always-neural relL2 = 0.2550 ± 0.0207)

| BRACE budget | relL2 | CI95 | Error reduction vs neural |
|---|---|---|---|
| 5% | 0.2217 | 0.0175 | 13.0% |
| 10% | 0.1970 | 0.0162 | 22.7% |
| 20% | 0.1588 | 0.0152 | 37.7% |
| 40% | 0.1083 | 0.0128 | 57.5% |
| 60% | 0.0645 | 0.0088 | 74.7% |

Baselines:
- Always-neural (A): 0.2550 ± 0.0207, cost 8.5 ms
- Always-numerical (B): 0.0000 (degenerate — same solver as reference), cost 11.0 ms
- Oracle (H): 0.0042 ± 0.0004, cost 21.7 ms, 95.5% intervention

### Budget enforcement (key differentiator)

- **G (BRACE):** actual intervention fraction = nominal budget exactly at all levels (0.05 → 0.050, ..., 0.60 → 0.600).
- **C/E/F (threshold/risk):** nominal 5% → actual ~66% intervention. They overshoot their declared budget under error accumulation.

### Cost accounting (all computation measured)

| Policy | neural ms | solver ms | controller ms | prepare ms (one-time) | total ms |
|---|---|---|---|---|---|
| A neural | 8.48 | 0 | 0.02 | 0 | 8.50 |
| B numerical | 0 | 10.96 | 0.02 | 0 | 10.98 |
| C fixed threshold | 0 | 7.42 | 13.45 | 0 | 20.87 |
| D adaptive threshold | 0 | 11.40 | 15.29 | 0 | 26.69 |
| E uncalibrated risk | 0 | 7.71 | 48.36 | 0 | 56.07 |
| F calibrated risk | 0 | 7.78 | 59.11 | 0 | 66.89 |
| G BRACE (0.2 budget) | 6.75 | 2.21 | 0.13 | 54.02 | 9.09 |

Note: E/F pay high per-step risk-compute cost (controller 48–59 ms) with accuracy identical to C (0.049) — dominated. G has low online cost (ctrl 0.13 ms) and a one-time prep cost (54 ms, reported separately).

### Research questions — answers

| RQ | Answer |
|---|---|
| RQ1: Risk features predict failure? | **Yes** — AUROC 0.988 ± 0.002 |
| RQ2: Calibration helps? | **No (null)** — ECE/Brier/AUROC identical; p=0.29 vs fixed threshold |
| RQ3: Risk-ranked budget allocation better? | **On budget discipline: yes.** G enforces budget exactly; thresholds overshoot. On raw accuracy at matched spend: no (thresholds more accurate but violate budget). |
| RQ4: Selective correction improves reliability? | **Yes** — G vs A: d=−1.45 to −2.03, p<1e-30 |
| RQ5: Accuracy-vs-cost relationship? | **Monotonic trade-off** — clear Pareto frontier, no dominant policy |
| RQ6: Robust to parameter/IC changes? | **Partially** — in-distribution robust; OOD (unseen ν) fail_rate ~0.67, AUROC ~0.90 |

### Statistical significance (paired, identical trajectories)

- G(0.2) vs A: mean diff −0.096, CI [−0.108, −0.085], d=−1.45, p=1.8e-32 ***
- G(0.4) vs A: d=−1.80, p=8.4e-41 ***
- G(0.6) vs A: d=−2.03, p=6.7e-46 ***
- F (calibrated) vs C (fixed): p=0.29, ns — calibration null confirmed

### Novelty (honest)

Not the first neural PDE surrogate, calibration method, or selective-prediction method. Defensible contribution: calibrated failure-probability estimation + budget-constrained risk-ranked selective numerical correction for lightweight CPU surrogates, with real wall-clock cost accounting and honest budget-enforcement finding.

### Reproducibility

- Inference from fixed checkpoint: deterministic.
- Retraining: moderate variation (PyTorch CPU nondeterminism) — report mean±std across seeds.
- Dataset + analysis: reproducible with checksums.

### Key limitations

- Single primary PDE (Burgers'); advection-diffusion secondary.
- Calibration assumes exchangeability; no distribution-free guarantees under shift.
- τ = 0.01 set a priori.
- CPU-only, 8 GiB RAM; small surrogate (~17K params).
- G's one-time prep cost (~54 ms) reported separately.
- OOD performance degrades (fail_rate ~0.67).

---

*End of final results. See docs/RESEARCH_FINDINGS.md for full detail and docs/LIMITATIONS.md for limitations.*
