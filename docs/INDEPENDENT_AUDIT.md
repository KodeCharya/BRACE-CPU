# Independent Scientific Audit — BRACE-CPU

**Date:** 2026-11-05  
**Scope:** Full end-to-end audit of source code, data, experiments, results, and statistics.

---

## Executive Summary

The BRACE-CPU project has been implemented end-to-end and experimentally validated. An independent audit identified **three defects**, all of which were fixed and the affected experiments rerun:

1. **Off-by-one in `times[]`** (`src/controller/rollout.py`): `times[i+1] = t` should be `t + dt`, corrupting the reference-lookup accuracy measurement. Fixed. All accuracy numbers in docs/RESEARCH_FINDINGS.md are from the corrected rerun.
2. **Uncosted controller/risk computation** (defects CF-1/CF-2): `decide()` called `surrogate.predict()` internally without timing it, and risk-based policies double-computed the neural prediction. Fixed by making `decide()` return `(intervene, u_pred, elapsed)` and timing controller time; G's one-time prep cost now measured separately.
3. **First main-run results superseded:** The original main run (exp_20261105_132231) used the defective code and is retained but superseded by the corrected rerun (exp_20261105_134910).

No data leakage was found (see docs/LEAKAGE_AUDIT.md). No financial/positive result was manufactured; null results (calibration, B-as-reference) are reported honestly.

---

## A. Experiment audit — complete

All components audited: source code, research protocol, dataset metadata, checksums, experiment configs, raw results, checkpoints, evaluation scripts, ablation outputs, statistical analysis.

Key traceability:
- Every reported finding traces to `results/raw/exp_20261105_134910_63a5fa94_all.json` (3,500 records).
- Summary CSVs: `results/processed/summary_by_policy.csv`, `G_brace_budget_by_budget.csv`.
- Figures: `results/processed/figures/pareto_accuracy_vs_intervention.png`, `brace_accuracy_vs_budget.png`.
- Dataset: `data/processed/burgers/` (675 trajectories, SHA-256 verified, no partition overlap).

---

## B. Information leakage audit — PASS

See docs/LEAKAGE_AUDIT.md. Verified:
1. Train/val/cal/test/ood partitions disjoint (675 unique trajectories, no overlap). ✅
2. Calibration data never enters model training. ✅
3. Test labels never influence risk calibration/Platt scaling. ✅
4. Operational controllers (A–G) never access future ground truth; only H (oracle) does, explicitly evaluation-only. ✅
5. No future information in autoregressive prediction. ✅

---

## C. Computational fairness audit — DEFECTS FIXED

See docs/COMPUTATIONAL_FAIRNESS_AUDIT.md. Fixed:
- CF-1: Controller/risk computation now timed (controller_wall_s).
- CF-2: `decide()` returns u_pred to avoid double neural compute.
- G's one-time prep cost measured separately (prepare_wall_s).

After fix, E/F revealed as expensive (56–67ms/rollout, dominated by per-step risk computation) with no accuracy benefit over C — dominated, not superior.

---

## D. Statistical audit — complete

See docs/STATISTICAL_AUDIT.md. Key verified results:
- G vs A: highly significant, very large effects (d = −1.45 to −2.03, p < 1e-30).
- Calibration (F vs C): no significant difference (p = 0.29, d = −0.10) — null result confirmed.
- Paired tests on identical (seed, trajectory) pairs; no pseudoreplication.
- "~50% error reduction at ~40% budget" verified (57.5% at budget 0.4).

---

## E. Risk estimator audit

- AUROC ≈ 0.988 in-distribution (robust across 5 seeds, std 0.0021).
- Feature ablation: `norm_residual` dominant (0.979); `mass_deviation` alone near-random (0.58); features redundant.
- OOD: AUROC degrades to ~0.90; failure rate jumps to ~0.67 (vs 0.09 in-dist).
- Precision/recall at 0.5 threshold: P ≈ 0.88, R ≈ 0.73 (in-distribution).

---

## F. Calibration audit — null result retained

- Calibration does not improve ECE (0.0096 vs 0.0103), Brier (0.0251 vs 0.0251), or AUROC (identical).
- Robust to calibration-set size (25%/50%/100%).
- Paired F vs C: p = 0.29, ns.
- Calibration adds per-step cost with no benefit. Retained as honest null finding.

---

## G. Controller and budget audit

- G (BRACE) enforces budget exactly (actual = nominal at 0.05/0.10/0.20/0.40/0.60).
- C/E/F overshoot nominal budget (target 0.05 → actual ~0.66) under error accumulation.
- Budget enforcement maintained under long-horizon error accumulation for G (pre-selected top-k).
- Intervention-count budget and measured wall-clock budget reported separately.

---

## H. OOD robustness audit

- OOD (ν = 0.005–0.0079, below train range 0.01–0.03; Gaussian-random-field ICs): fail_rate ~0.67, mean_err ~0.014, AUROC ~0.90.
- Genuine extrapolation (unseen low-ν) is where BRACE becomes less reliable.
- Interpolation (in-distribution) robust.

---

## I. Reproducibility audit — partial

See docs/REPRODUCIBILITY_AUDIT.md.
- Inference from fixed checkpoint: deterministic, bitwise-identical.
- Retraining: moderate variation (PyTorch CPU training nondeterminism). Report mean±std across 5 seeds.
- Dataset and analysis: reproducible.

---

## J. Final claim classification

| Claim | Classification | Evidence |
|---|---|---|
| Risk features predict surrogate failure (RQ1) | **Supported** | AUROC 0.988 ± 0.002, 5 seeds |
| Calibration improves reliability (RQ2) | **Unsupported (null)** | p=0.29, ECE/Brier/AUROC identical |
| Risk-ranked intervention enforces budget better than thresholds (RQ3) | **Supported** | G exact; C/E/F overshoot to ~0.66 |
| Selective correction improves long-horizon reliability (RQ4) | **Supported** | G vs A: d=−1.45 to −2.03, p<1e-30 |
| Monotonic accuracy-vs-cost trade-off (RQ5) | **Supported** | relL2 decreases monotonically with budget |
| Robust to parameter/IC changes (RQ6) | **Partially supported** | In-distribution robust; OOD degrades (fail_rate ~0.67) |
| ~50% error reduction at ~40% budget | **Supported** | 57.5% reduction at budget 0.4 |
| Calibration adds value | **Unsupported (null)** | F dominated by C (same accuracy, higher cost) |

---

## K. Final decision

**Audit PASSES with corrections applied.** All defects fixed, affected experiments rerun, null results honestly reported. The project is ready for manuscript finalization (Phase 13). No external submission without explicit authorization.

---

*End of independent audit.*
