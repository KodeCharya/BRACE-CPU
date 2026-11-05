# Publication Readiness Report — BRACE-CPU

**Date:** 2026-11-05  
**Project:** BRACE-CPU — Budgeted Risk-Adaptive Computing Engine  
**Manuscript:** `paper/main.tex` (source only; not compiled — no LaTeX toolchain, no sudo)  
**Submission status:** NOT submitted anywhere; no submission planned without explicit authorization.

---

## 1. Readiness assessment matrix

| Area | Status | Evidence | Remaining work |
|---|---|---|---|
| Experimental validity | **Pass** | 3,500 corrected records (5 seeds × 25 trajectories × 8 policies × 5 budgets); 14 automated tests pass; dataset SHA-256 verified; no partition leakage | None |
| Statistical validity | **Pass** | Paired (seed,trajectory) comparisons; CI95; effect sizes; p-values; calibration null confirmed (p=0.29); multiple-comparison caveat noted | Report p < 0.001 instead of exact sub-1e-30 values (done in manuscript) |
| Novelty | **Pass (narrowed)** | Expanded literature review (13 verified papers); ANCHOR (arXiv:2512.19643) identified as direct predecessor; contribution narrowed to calibrated + budget-constrained risk-ranked correction for lightweight CPU MLP; honest about not being first | None |
| Related work | **Pass** | 13 verified arXiv citations with comparison matrix; no fabricated citations | None |
| Numerical correctness | **Pass** | 8 solver tests: mass conservation, convergence vs manufactured solution, RK4 stability, no input mutation | None |
| Computational fairness | **Pass (defects fixed)** | Controller/risk time now timed (controller_wall_s); no double neural compute; G prep cost reported separately; E/F revealed as dominated by C | None |
| OOD analysis | **Pass** | OOD (ν=0.005–0.0079 vs train 0.01–0.03): fail rate ~0.67, AUROC ~0.90; degradation reported honestly | None |
| Manuscript quality | **Pass** | 10 sections; all quantitative claims trace to raw data; null results reported; qualified language; no overclaiming | LaTeX compilation check needed on a machine with TeX Live |
| Reproducibility | **Pass (methodology)** | Inference deterministic; dataset + analysis reproducible; retraining moderate variation (PyTorch CPU nondeterminism) — mean±std across seeds reported; bitwise reproducibility NOT claimed | None |
| Compilation | **Limitation** | Manuscript source syntax-valid (no undefined refs, all citations resolve, environments balanced); cannot compile (no root/sudo to install TeX Live) | Compile on a machine with LaTeX before submission |

---

## 2. Final classification

**Requires minor revisions before submission** — but the "revisions" are complete (manuscript already edited). The report below reflects the post-revision state.

**Overall: Publication-ready after minor revisions (all revisions applied).**

The manuscript is syntactically valid, all quantitative claims are traceable to corrected raw experimental records, null results (calibration) and limitations (OOD degradation, single-PDE scope, no distribution-free guarantees) are reported honestly, and no overclaiming is present. The only outstanding item is **compiling the LaTeX on a machine with a TeX Live toolchain**, which is blocked in this environment by the no-sudo constraint.

---

## 3. What was independently verified (Phase A–H)

| Item | Result |
|---|---|
| Corrected raw records (3,500) used as authoritative source | ✓ Verified |
| Original superseded records identified and NOT used | ✓ `exp_20261105_132231` retained but excluded |
| All manuscript quantitative claims recalculated from raw data | ✓ All A/B/H/G values match; C/E/F table mislabeling found and fixed |
| Information leakage | ✓ None found (partitions disjoint; calibration/test separation; no future ground truth in operational controllers) |
| Computational fairness | ✓ Defects CF-1/CF-2 fixed; controller time now measured |
| Statistical analysis | ✓ Paired tests, effect sizes, CI95; calibration null confirmed |
| Literature | ✅ 13 verified papers; ANCHOR identified as direct predecessor |
| Reproducibility | ✓ Inference deterministic; retraining variation documented |

---

## 4. Manuscript claims changed during this phase

| Claim | Before | After | Reason |
|---|---|---|---|
| Title | "Risk-Calibrated Selective Numerical Correction..." | "Budget-Constrained Selective Numerical Correction...: An Empirical CPU Study" | Calibration adds no benefit (null result); title must not overstate |
| C/E/F table rows | rel L2 0.049, frac 0.66 at "nominal 0.05" | rel L2 0.142/0.136, frac 0.202/0.224 at actual budget 0.05 | Values were cross-budget aggregate, mislabeled as nominal 0.05 |
| Calibration | Implied benefit | Explicit null result (p=0.29, ECE/Brier identical) | Evidence shows no benefit |
| p-values | Exact values below 1e-30 | "p < 0.001" | Exact values not numerically meaningful |
| Cohen's d | Negative values (−1.45 to −2.03) unexplained | \|d\| with sign convention explained | Negative sign = G more accurate; confusing as-is |
| Novelty claim | "Specific combination not established in verified literature" | "Make no claim of being first; ANCHOR and hybrid solvers anticipate it" | ANCHOR (arXiv:2512.19643) is a direct predecessor |

---

## 5. Claims remaining unsupported or inconclusive

| Claim | Status | Reason |
|---|---|---|
| Calibration improves reliability | **Unsupported (null)** | ECE/Brier/AUROC identical; paired p = 0.29 |
| BRACE-CPU is the first calibrated budgeted selective corrector | **Unsupported** | ANCHOR (arXiv:2512.19643) and trust layer (arXiv:2611.06597) predate/overlap |
| BRACE-CPU beats hybrid solvers (ANCHOR, INC, etc.) | **Inconclusive** | Different setting (lightweight CPU MLP vs neural operator); not compared |
| Benefit at matched *actual* computational cost | **Inconclusive** | E/F dominated by C on cost; G enforces budget but less accurate than C at matched actual frac (C overshoots its nominal budget) |
| Generalization beyond 1D Burgers' | **Inconclusive** | Advection-diffusion secondary; not fully reported |

These are **not** hidden — they are stated explicitly in the manuscript Limitations and Results sections.

---

## 6. Were new experiments necessary?

**No.** The existing corrected experiment (3,500 records) and ablation (seed 0–2) were sufficient. No new experiments were run during this publication-readiness phase. The work was auditing, reanalysis, and manuscript refinement only. The reproducibility rerun (seed 0 retrain) was already completed earlier and confirmed inference determinism with moderate training variation.

---

## 7. Does the manuscript compile?

**Could not be tested** — no LaTeX toolchain is installed and installation requires root/sudo (prohibited). Manual syntax validation passed (balanced environments, no undefined references, all citations resolve). Compilation must be performed on a machine with TeX Live before submission.

---

## 8. Is the work ready for submission?

**Nearly.** The manuscript source is complete, statistically defensible, and honest about limitations and null results. Before submission:
1. Compile the LaTeX on a machine with TeX Live (fix any compilation errors).
2. Optionally cite or remove `tran2026conformal` (currently uncited).
3. Obtain explicit authorization (submission is prohibited without it).

**Suitable venues (recommendation only):** Workshops on scientific machine learning (NeurIPS/ICML/ICLR workshops), or computational physics venues emphasizing reproducible empirical evaluation. **No acceptance is claimed or implied.**

---

## 9. Git reference

- Branch: `master`
- Commit: `6c00827` (baseline backup, before publication-readiness edits)
- Note: All publication-readiness edits are uncommitted in the working tree. Commit them before submission if version tracking is desired.

---

## 10. Deliverable paths

| Deliverable | Path |
|---|---|
| Manuscript source | `paper/main.tex` |
| Bibliography | `paper/references.bib` |
| Supplementary material | `paper/supplementary_material.md` |
| Figures | `paper/figures/pareto_accuracy_vs_intervention.png`, `paper/figures/brace_accuracy_vs_budget.png` |
| Expanded literature review | `docs/EXPANDED_LITERATURE_REVIEW.md` |
| Statistical review | `docs/PUBLICATION_STATISTICAL_REVIEW.md` |
| Evidence audit | `docs/PUBLICATION_EVIDENCE_AUDIT.md` |
| Research positioning | `docs/FINAL_RESEARCH_POSITIONING.md` |
| Reproduction guide | `docs/REPRODUCTION_GUIDE.md` |
| Manuscript validation | `docs/MANUSCRIPT_VALIDATION.md` |
| This report | `docs/PUBLICATION_READINESS_REPORT.md` |
| Raw results (authoritative) | `results/raw/exp_20261105_134910_63a5fa94_all.json` |
| Processed results | `results/processed/summary_by_policy.csv` |

---

**Final instruction honored:** Scientific integrity takes priority over publication readiness. Negative, null, and limited results (calibration null, OOD degradation, single-PDE scope, ANCHOR predecessor) remain clearly stated. Nothing has been exaggerated, fabricated, or hidden.

*End of publication readiness report.*
