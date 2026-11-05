# Project Status — BRACE-CPU

**Current phase:** ALL PHASES COMPLETE (0–13 + publication-readiness validation A–I)  
**Git:** branch `master`, commit `9e770df` (publication-readiness validation)  
**Last updated:** 2026-11-05  
**Submission status:** NOT submitted; not to be submitted without explicit authorization

## Completed (all phases)

Phase 0 Environment audit; Phase 1 Literature review (5 verified papers); Phase 2 Research protocol; Phase 3 Solvers (8 tests pass); Phase 4 Dataset (675 trajectories, SHA-256 verified, no partition leakage); Phase 5 Surrogate (17K-param MLP, increment formulation); Phase 6 Risk+calibration (AUROC 0.988, calibration null); Phase 7 Controller (policies A–H, wall-clock cost accounting); Phase 8 Pilot (1,260 records); Phase 9 Main corrected rerun (3,500 records, exp_20261105_134910_63a5fa94); Phase 10 Ablations+robustness; Phase 11 Statistical analysis; Phase 12 Reproducibility audit; Phase 13 Manuscript source; Publication-readiness validation A–I.

## Key results (corrected, 5 seeds, mean ± 95% CI)
- A neural: relL2 0.2550 ± 0.0207, cost 8.5 ms, frac 0.000
- G BRACE: 0.222 (5%) → 0.065 (60%); enforces budget exactly; ~57.5% error reduction at 40% budget
- B numerical: 0.000 (degenerate—same solver as reference), cost 11.0 ms, frac 1.000
- H oracle: 0.0042 ± 0.0004, cost 21.7 ms, frac 0.955
- C/E/F at nominal 0.05: actual frac 0.202/0.224/0.224 (OVERSHOOT budget); relL2 0.142/0.136/0.136
- Calibration: null result (ECE 0.0096 vs 0.0103; Brier/AUROC identical; p=0.29)
- Paired: G vs A |d|=1.45–2.03, p<0.001; F vs C p=0.29 (ns)
- OOD: fail rate ~0.67 (vs ~0.09 in-dist), AUROC ~0.90

## Claim classification
Supported: RQ1, RQ3(budget discipline), RQ4, RQ5, ~50% error reduction
Unsupported/null: RQ2 (calibration), calibration value
Partially supported: RQ6 (robust in-distribution, degrades OOD)

## Defects found & fixed (before corrected rerun)
CF-1: uncosted controller/risk computation → now timed; CF-2: double neural compute → decide() returns u_pred; times[] off-by-one → t+dt; duplicate Conclusion section; C/E/F table mislabeling; exact sub-1e-30 p-values; negative Cohen's d sign.

## Manuscript status
paper/main.tex (complete, syntax-valid, all citations resolve, no undefined refs); paper/references.bib (13 verified citations); paper/figures/ (2 figures); paper/supplementary_material.md. NOT compiled (no root/sudo to install TeX Live). NOT submitted.

## Resource consumption
Within 8 GiB RAM / 4-core CPU limits throughout. Sequential execution. ~87 GiB disk available.

## Next / final
Project complete. Honest publication-readiness: manuscript source complete and validated; results audit-passed; compile LaTeX on a TeX Live machine before submission; obtain explicit authorization before any submission.
