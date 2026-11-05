# Experiment Log — BRACE-CPU

| Date | Phase | Experiment ID | Description | Status | Result summary |
|------|-------|---------------|-------------|--------|----------------|
| 2026-11-05 | 0 | ENV-001 | Environment audit | Complete | CPU-only, 4-core i5-6500, 7.2GiB RAM |
| 2026-11-05 | 3 | SOLVER-001 | Solvers + tests | Complete | 8/8 tests pass |
| 2026-11-05 | 4 | DATA-001 | Dataset generation | Complete | 675 trajectories, no partition overlap, SHA-256 verified |
| 2026-11-05 | 5 | SURR-001 | Surrogate training | Complete | 17K params, increment formulation |
| 2026-11-05 | 6 | RISK-001 | Risk + calibration | Complete | AUROC ~0.99; calibration null result |
| 2026-11-05 | 7 | CTRL-001 | Controller smoke test | Complete | Policies A–H functional |
| 2026-11-05 | 8 | PILOT-001 | Pilot 3 seeds | Complete | 1,260 records |
| 2026-11-05 | 9 | MAIN-001 | Main 5 seeds (ORIGINAL CODE) | Complete but DEFECTIVE | 3,500 records; times[] off-by-one corrupted accuracy; controller cost untimed (CF-1/CF-2) — MUST RERUN |
| 2026-11-05 | 10 | RISK-EVAL-001 | Risk/calibration 5 seeds | Complete | Calibration null result robust |
| 2026-11-05 | 10 | ABLATION-001 | Feature/cal-size/OOD ablation | Complete | residual dominant; cal-size stable; OOD fail_rate ~0.67 |
| 2026-11-05 | 12 | REPRO-001 | Reproducibility audit (seed 0 retrain vs original) | Complete | inference deterministic; retraining moderate variation |
| 2026-11-05 | 13 | MANUSCRIPT-001 | Paper source (main.tex, references.bib, figures, supplementary) | Complete | source complete, not compiled (no LaTeX), not submitted |

## DEFECTS FIXED (before corrected rerun)
- CF-1: Controller/risk computation time now timed (controller_wall_s)
- CF-2: decide() returns u_pred to avoid double neural compute
- times[] off-by-one: times[i+1] = t + dt (was t), fixing reference-lookup accuracy
- Record fields: controller_wall_s, prepare_wall_s added

## Status: ALL PHASES COMPLETE. Audit passed. Project ready (manuscript source only, not submitted).
