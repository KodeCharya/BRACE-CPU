# Project Status — BRACE-CPU

**Current phase:** Phase 12 — Reproducibility audit (COMPLETE); Phase 13 — Manuscript (COMPLETE, source only)  
**Audit status:** PASSED (3 defects fixed + rerun; docs/INDEPENDENT_AUDIT.md)  
**Last updated:** 2026-11-05

## Completed work
All phases 0–13 complete:
- Phase 0: Environment audit (docs/ENVIRONMENT_REPORT.md)
- Phase 1: Literature review (docs/LITERATURE_REVIEW.md, NOVELTY_ANALYSIS.md, RELATED_WORK_MATRIX.csv) — 5 verified arXiv citations
- Phase 2: Research protocol (docs/RESEARCH_PROTOCOL.md)
- Phase 3: Numerical solvers — 8 tests pass
- Phase 4: Dataset — 675 trajectories, 5 partitions, SHA-256 verified, no partition overlap
- Phase 5: MLP surrogate — 17K params, increment formulation, beats identity baseline
- Phase 6: Risk + calibration — AUROC 0.988; calibration null result
- Phase 7: Controller policies A–H with wall-clock cost accounting
- Phase 8: Pilot (1,260 records)
- Phase 9: Main (3,500 records, corrected rerun exp_20261105_134910_63a5fa94)
- Phase 10: Ablation + robustness (docs/results/processed/ablation_output.txt)
- Phase 11: Statistical analysis (docs/STATISTICAL_AUDIT.md) — paired tests, effect sizes
- Phase 12: Reproducibility audit (docs/REPRODUCIBILITY_AUDIT.md) — inference deterministic; retraining moderate variation
- Phase 13: Manuscript — paper/main.tex, paper/references.bib (5 verified citations), paper/figures/, paper/supplementary_material.md

## Audit reports (new)
- docs/INDEPENDENT_AUDIT.md — overall audit, claim classifications
- docs/STATISTICAL_AUDIT.md — recalculated metrics, paired tests, effect sizes
- docs/LEAKAGE_AUDIT.md — no leakage found
- docs/REPRODUCIBILITY_AUDIT.md — reproducibility assessment
- docs/COMPUTATIONAL_FAIRNESS_AUDIT.md — defects CF-1/CF-2 found & fixed
- docs/FINAL_RESULTS.md — headline numbers

## Key results (corrected, 5 seeds)
- G BRACE vs A neural: d=−1.45 to −2.03, p<1e-30 (supported)
- Calibration (F vs C): p=0.29, ns (null, honestly reported)
- G enforces budget exactly; C/E/F overshoot to ~66% at nominal 5%
- ~50% error reduction at 40% budget (57.5% at budget 0.4)
- OOD degrades (fail_rate ~0.67)

## Claim classification (docs/INDEPENDENT_AUDIT.md)
Supported: RQ1, RQ3(budget), RQ4, RQ5, ~50% reduction
Unsupported/null: RQ2 (calibration), calibration value
Partially supported: RQ6 (robustness)

## Resource consumption
Within 8 GiB RAM / 4-core CPU limits throughout. Sequential execution. No GPU/cloud.

## Manuscript status
paper/main.tex (complete source), paper/references.bib (5 verified citations), paper/figures/ (2 figures), paper/supplementary_material.md. NOT compiled (no LaTeX installed). NOT submitted anywhere.

## Next / final
Project complete per completion criteria. Honest publication-readiness assessment: manuscript source complete; results audit-passed; would benefit from LaTeX compilation check and broader literature confirmation of novelty before any submission (submission not authorized).
