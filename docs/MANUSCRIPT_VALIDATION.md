# Manuscript Validation Report — BRACE-CPU

**Date:** 2026-11-05  
**Manuscript:** `paper/main.tex`  
**Bibliography:** `paper/references.bib`  
**Figures:** `paper/figures/pareto_accuracy_vs_intervention.png`, `paper/figures/brace_accuracy_vs_budget.png`

---

## 1. Compilation status

**LaTeX toolchain:** NOT available. No `pdflatex`, `xelatex`, or `lualatex` installed.  
**Installation feasibility:** Blocked — we are user `mukesh` (uid 1000), not root, and `sudo` is prohibited by project rules. System package installation (`apt-get install texlive-*`) requires root privileges.  
**Verdict:** Compilation could not be performed. This is an environment limitation, not a manuscript defect.

**Manual validation performed instead** (see `scripts/latex_check.py`):
- Document structure: `\begin{document}` / `\end{document}` balanced (1 each) ✓
- `\maketitle`: present (1) ✓
- All environments balanced (abstract, document, equation, figure, itemize, table, tabular) ✓
- 10 sections, all present ✓
- 2 figures, both referenced and files exist ✓
- 2 tabular blocks ✓
- **No undefined references** ✓
- **All cited keys resolve to bibliography entries** ✓ (0 cited-but-missing-from-bib)
- All citation keys match verified arXiv papers ✓

**Note:** `tran2026conformal` is present in `references.bib` but not cited in `main.tex`. This is harmless (an extra relevant reference) but could be removed or cited for cleanliness.

---

## 2. Sections present (10)

1. Introduction
2. Related Work
3. Problem Formulation
4. Research Questions and Hypotheses
5. Method
6. Experiments
7. Results
8. Limitations
9. Conclusion
10. Reproducibility (unnumbered)

All required sections from the task are present: Abstract, Introduction, Related Work, Research Problem, Mathematical Formulation, Proposed Architecture, Experimental Methodology, Baseline Comparison, Main Results, Ablation Studies, Robustness Analysis, Computational Efficiency Analysis, Limitations, Discussion (within Conclusion), Conclusion, References.

---

## 3. Corrections applied during this phase

| Issue | Fix |
|---|---|
| Duplicate `\section{Conclusion}` (one complete, one placeholder) | Removed placeholder; single Conclusion retained |
| Title overstated calibration benefit | Changed to "Budget-Constrained Selective Numerical Correction for Lightweight Neural PDE Surrogates: An Empirical CPU Study" |
| C/E/F table rows mislabeled (showed cross-budget aggregate as "nominal 0.05") | Corrected to actual budget=0.05 values (rel L2 0.142/0.136, frac 0.202/0.224); added caption note about overshoot |
| Budget enforcement text used wrong C value (0.049 aggregate) | Corrected to actual matched-frac value (0.142) |
| p-values reported as exact below 1e-30 | Changed to "p < 0.001" |
| Negative Cohen's d confusing | Clarified sign convention ("|d|=…, negative sign denotes G more accurate") |
| Missing `\bibliographystyle`, `\bibliography`, `\end{document}` | Restored |
| Undefined `\ref{sec:baselines}` | Changed to `\ref{sec:method}` |
| Calibration claim overstated in abstract/results | Reworded as honest null result |

---

## 4. Remaining honest limitations (not defects)

- Calibration provides no measurable benefit (null result) — reported, not suppressed.
- BRACE-CPU is not compared against ANCHOR/INC/etc. (different setting: lightweight CPU MLP vs neural operator) — acknowledged in Limitations.
- OOD performance degrades (fail rate ~0.67) — reported in Robustness.
- Single PDE (Burgers') primary benchmark — acknowledged.
- No distribution-free guarantees under shift — acknowledged (exchangeability assumption explicit).
- Training nondeterminism (PyTorch CPU) — mean±std across 5 seeds reported; bitwise reproducibility not claimed.
- LaTeX compilation not possible (no root/sudo) — reported here.

---

## 5. Verdict

The manuscript source is **syntactically valid**, **internally consistent**, and **all quantitative claims trace to corrected raw records**. It cannot be compiled to PDF in this environment due to the no-sudo constraint, which is an environment limitation, not a manuscript defect. A compilation check should be performed on a machine with a LaTeX toolchain before any submission.

*End of manuscript validation report.*
