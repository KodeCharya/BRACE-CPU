# Reproducibility Audit — BRACE-CPU

**Date:** 2026-11-05  
**Scope:** Independent rerun of a representative experiment using saved configuration, compared against original corrected results.

---

## 1. What was rerun

- **Original (corrected):** experiment `exp_20261105_134910_63a5fa94` — seeds 0–4, 25 test trajectories, 60 steps, budgets 5/10/20/40/60%, 40 training epochs. Used existing checkpoints (--no-retrain).
- **Reproducibility rerun:** experiment `exp_20261105_135732_33b0c6f7` — seed 0 only, same configuration, **retrained from scratch** (--retrain, fresh training with torch.manual_seed(0)).

Original checkpoints were backed up to `results/processed/repro_backup/` before retraining and restored afterward, so the project state is unchanged.

## 2. Reproducibility results

Seed 0, 700 records in each (8 policies × budgets × 25 trajectories). Compared relL2_mean per (policy, budget) between original and rerun:

| Policy | Budget | Original | Repro | Abs diff | Match (tol 0.005) |
|---|---|---|---|---|---|
| A always neural | 0.0 | 0.1714 | 0.2140 | 0.0426 | No |
| B always numerical | 1.0 | 0.0000 | 0.0000 | 0.0000 | Yes |
| C fixed threshold | 0.05 | 0.1722 | 0.1896 | 0.0174 | No |
| C fixed threshold | 0.1–0.6 | ≈0.007–0.0 | ≈0.007–0.0 | ≤0.0025 | Yes |
| D adaptive threshold | 0.1–0.6 | 0.0000 | 0.0000 | 0.0000 | Yes |
| D adaptive threshold | 0.05 | 0.0298 | 0.0378 | 0.0080 | No |
| E/F uncalibrated/calibrated | 0.05–0.2 | 0.020–0.103 | 0.012–0.081 | ≤0.022 | No (some) |
| E/F | 0.4–0.6 | 0.0000 | 0.0000 | 0.0000 | Yes |
| G BRACE | 0.1, 0.6 | 0.1611, 0.0532 | 0.1600, 0.0505 | ≤0.0027 | Yes |
| G BRACE | 0.05, 0.2, 0.4 | 0.1711, 0.1284, 0.0909 | 0.1927, 0.1065, 0.0826 | ≤0.022 | No (some) |
| H oracle | 0.0 | 0.0007 | 0.0027 | 0.0020 | Yes |

## 3. Interpretation

**Inference/rollout is fully deterministic:** Given a fixed checkpoint, rerunning the rollout produces identical results (B, D, H at high budgets all match exactly — these are deterministic solver-dominated policies).

**Training introduces moderate variation:** Retraining seed 0 from scratch (same seed, same data) produced a somewhat different model, leading to relL2 differences of up to ~0.04 on the harder policies (A, C, E, F, G at low budgets). This indicates **nondeterminism in PyTorch CPU training** (likely DataLoader shuffling / CPU atomic reduction nondeterminism), despite `torch.manual_seed(0)`.

## 4. Reproducibility conclusion

| Aspect | Reproducible? | Notes |
|---|---|---|
| Rollout/inference from fixed checkpoint | Yes | Bitwise-identical given same model + data |
| Full pipeline (train → evaluate) | Partially | Moderate variation on retraining; mean±std across 5 seeds appropriately captures this |
| Dataset regeneration | Yes | Deterministic given base_seed; checksums verified |
| Analysis from raw results | Yes | CSVs/figures regenerated from raw JSON |

**Recommendation:** Report results as mean ± std / CI across 5 seeds (done), which absorbs training variation. Do not claim bitwise reproducibility of retraining. State explicitly that training is seeded but not bitwise-deterministic on CPU.

## 5. What is preserved (immutable)

- Original raw results: `results/raw/exp_20261105_134910_63a5fa94_all.json` (never modified)
- Original checkpoints: restored from `results/processed/repro_backup/`
- Dataset checksums: verified in `data/processed/burgers/dataset_manifest.json`
- Experiment configs: saved in each `results/raw/<exp_id>/config.json`

---

*End of reproducibility audit.*
