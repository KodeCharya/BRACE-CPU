# Project Status — BRACE-CPU

**Current phase:** Publication-readiness validation (Phases A–H) — IN PROGRESS  
**Sub-phase:** Phase A — Evidence audit (COMPLETE); Phase B — Statistical review (COMPLETE); Phase C — Literature expansion (IN PROGRESS)  
**Last updated:** 2026-11-05  
**Git:** branch `master`, commit `6c00827` (baseline backup committed)

## Completed
- Phase A: docs/PUBLICATION_EVIDENCE_AUDIT.md — all claims traced to corrected raw records; 3 discrepancies found (D1–D5)
- Phase B: docs/PUBLICATION_STATISTICAL_REVIEW.md — full recalculation; paired tests; effect sizes; calibration null confirmed
- Git baseline commit `6c00827` created as backup before modifications

## Key defects found in manuscript
- D1: C/E/F table rows labeled "nominal 0.05" but show cross-budget aggregate (relL2 0.049, frac 0.66). At actual budget 0.05: relL2 0.142, frac 0.20. MISLABELED.
- D2: Text claims C better than G at matched ~20% frac using 0.049 (aggregate) — wrong; actual matched-frac C = 0.142 vs G 0.159.
- D3: Exact p-values below 1e-30 not numerically meaningful; report as p < 0.001.
- D4: Negative Cohen's d confusing; clarify sign convention.
- D5: Duplicate `\section{Conclusion}` (complete + placeholder) — remove placeholder.
- Manuscript missing: explicit hypotheses (H0/H1), numerical solver details, threat-to-validity section, several required figures/tables.

## Remaining
- Phase C: Expanded literature review (docs/EXPANDED_LITERATURE_REVIEW.md)
- Phase D: Research positioning (docs/FINAL_RESEARCH_POSITIONING.md) — title question (calibration adds no benefit)
- Phase E: Manuscript refinement (fix D1–D5, add missing sections, qualified language)
- Phase F: Figures/tables audit + generation from raw data
- Phase G: Reproduction guide (docs/REPRODUCTION_GUIDE.md)
- Phase H: LaTeX compilation check + docs/MANUSCRIPT_VALIDATION.md
- Phase I: Final publication-readiness report (docs/PUBLICATION_READINESS_REPORT.md)

## Next action
Phase C: Write expanded literature review covering hybrid neural-numerical solvers, adaptive correction, learned error indicators, selective prediction, risk-aware inference, budget-constrained computation, conformal prediction in SciML, OOD reliability.
