# Information Leakage Audit — BRACE-CPU

**Date:** 2026-11-05  
**Scope:** All source code, dataset generation, experiment harness, risk estimation, calibration, and controller logic.

---

## 1. Data partition integrity (VERIFIED)

Dataset: `data/processed/burgers/dataset_manifest.json`, 675 trajectories.

| Partition | Count |
|---|---|
| train | 300 |
| validation | 75 |
| calibration | 75 |
| test | 150 |
| ood_test | 75 |
| **Total** | **675** |

- All 675 trajectory IDs are **unique** (no duplicates).
- **No overlap** between any pair of partitions (train/validation/calibration/test/ood_test).
- Each `.npz` file has a matching `.json` metadata file with a verified SHA-256.
- Manifest SHA-256 is recorded.

**Conclusion:** No training/validation/calibration/test trajectory leakage. ✅

## 2. Calibration data vs model training (VERIFIED SEPARATION)

- Surrogate is trained on `train` partition only (with `validation` for early-stopping model selection).
- Risk estimator is fit on `calibration` partition only.
- Test and ood_test partitions are **never** used for training, normalization-stat fitting, risk-model fitting, or Platt scaling.
- Normalization statistics (`u_mean`, `u_std`, etc.) are computed from `train` states only.

**Conclusion:** Calibration data never enters model training. ✅

## 3. Test labels never influence risk calibration (VERIFIED)

- `RiskEstimator.fit()` uses only calibration trajectories. It computes `u_pred = surrogate.predict(...)` and compares to `u_true` **within calibration trajectories only** to build labels `y = (err > tau)`.
- Test trajectories are used only for final evaluation (AUROC, ECE, policy accuracy), never to fit `risk_model` or `platt`.

**Conclusion:** Test labels never influence calibration. ✅

## 4. Operational controllers never access future ground truth (VERIFIED)

Reviewed all policy `decide()` paths:

| Policy | Future ground truth accessed? | Notes |
|---|---|---|
| A always_neural | No | Pure surrogate inference |
| B always_numerical | No | Solver from current state only |
| C fixed_threshold | No | Residual from `u_t`, `u_pred` only |
| D adaptive_threshold | No | Residual + history only |
| E uncalibrated_risk | No | Risk features from `u_t`, `u_pred` only |
| F calibrated_risk | No | Calibrated risk from features only |
| G budget_risk_ranked | **Partially** | Pass-1 uses an **all-surrogate trajectory** (no ground truth) to pre-score risk. Safe — no true future used. Selects top-k step indices from surrogate-only rollout. |
| H oracle | **Yes** | Uses `reference_states`/`reference_times` (ground truth) — explicitly marked "evaluation only", never available to the operational controller. |

**G pass-1 detail:** `_prepare_budget_ranked` runs the surrogate autoregressively (`u = u_pred`) to score risk at each step. It does **not** inject the true state. This is safe (no leakage) but is a design approximation: the risk scores are computed along a surrogate-only trajectory, while the actual controlled rollout may diverge once interventions begin. This is a **known limitation**, not a leakage.

**Conclusion:** Operational controllers (A–G) never access future ground truth. Only H (oracle) does, and it is explicitly evaluation-only. ✅

## 5. No future trajectory information in autoregressive prediction (VERIFIED)

- `NeuralSurrogate.predict(u, params, t)` takes only current state `u`, parameters `params`, and current time `t`.
- The increment target in training is `u_{t+dt} - u_t`, computed from consecutive states within the **same** trajectory — this is standard supervised next-step prediction, not lookahead.
- Autoregressive rollout feeds each prediction back as the next input (`u = u_pred`). No teacher-forcing with ground truth at rollout time.

**Conclusion:** No future information enters prediction. ✅

---

## 6. Potential leakage paths examined and dispositioned

| Potential path | Examined? | Disposition |
|---|---|---|
| Normalization stats from test | Yes | Computed from train only |
| Validation used for model selection | Yes | `best_state` selected by val loss — standard, not test leakage |
| Risk-model labels from test | Yes | Labels from calibration only |
| Platt scaling on test | Yes | Platt fit on calibration half; evaluated on test |
| Controller `prepare()` sees reference | Yes | Only G (surrogate-only pass) and H (ground truth, eval-only) use reference |
| OOD trajectories in train | No — OOD nu (0.005–0.0079) is outside train range (0.01–0.03), genuine extrapolation |

---

## 7. Residual design concern (not leakage, but a validity issue)

**G's pass-1 risk scoring uses an all-surrogate trajectory.** Once interventions begin in the controlled rollout, the state diverges from the all-surrogate trajectory used for pre-scoring. So the pre-selected "top-k riskiest" indices are computed under a different state distribution than the one actually encountered. This is an **approximation**, not a data leak, but it means G's intervention positions are not perfectly adapted to the actual controlled trajectory. This should be acknowledged as a limitation.

---

*End of leakage audit.*
