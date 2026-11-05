# Final Research Positioning — BRACE-CPU

**Date:** 2026-11-05  
**Basis:** Expanded literature review (docs/EXPANDED_LITERATURE_REVIEW.md), evidence audit (docs/PUBLICATION_EVIDENCE_AUDIT.md), statistical review (docs/PUBLICATION_STATISTICAL_REVIEW.md).

---

## 1. Title assessment

**Current title:** *Risk-Calibrated Selective Numerical Correction for Lightweight Neural PDE Surrogates.*

**Problem:** The experimental evidence shows calibration (Platt scaling) provides **no measurable benefit** (ECE 0.0096 vs 0.0103, Brier identical, AUROC identical, paired p = 0.29). The word "Risk-Calibrated" overstates what was demonstrated. A more accurate title would de-emphasize or remove "Calibrated."

**Candidate titles:**
1. (Current) *Risk-Calibrated Selective Numerical Correction for Lightweight Neural PDE Surrogates.* — Overstates calibration benefit; not recommended.
2. *Budget-Constrained Selective Numerical Correction for Lightweight Neural PDE Surrogates: An Empirical CPU Study.* — Accurate; emphasizes budget enforcement (the supported result) and drops the unsupported calibration claim.
3. *Risk-Informed Selective Numerical Correction for Lightweight Neural PDE Surrogates under a Computational Budget.* — Accurate; "risk-informed" is broader than "risk-calibrated."

**Decision:** Use **option 2** (*Budget-Constrained Selective Numerical Correction for Lightweight Neural PDE Surrogates: An Empirical CPU Study.*) as the primary title. It:
- Reflects the strongest supported result (budget enforcement; G enforces budget exactly, thresholds overshoot).
- Drops "calibrated" (null result).
- Keeps "selective numerical correction" (supported).
- Keeps "lightweight neural PDE surrogates" (supported: 17K-param MLP).
- Adds "empirical CPU study" (honest about scope/constraints).
- Avoids "first-ever," "state-of-the-art," "guaranteed," "optimal."

If the venue prefers emphasis on risk, option 3 is an acceptable alternative. Option 1 should NOT be used.

---

## 2. Central contribution (supported only)

BRACE-CPU's defensible contribution is **not** "the first calibrated selective numerical corrector." ANCHOR (Roy et al., arXiv:2512.19643) and the trust layer (Vakhshouri, arXiv:2611.06597) and hybrid solvers (INC, HIN-LRI, MC², ENS) predate or overlap it.

The defensible contribution is:

> An empirical, CPU-constrained investigation showing that:
> 1. **Budget-constrained, risk-ranked selective numerical correction enforces a declared computational budget exactly**, whereas fixed-threshold and risk-threshold policies overshoot their nominal budget under long-horizon error accumulation.
> 2. **Calibrated failure-probability estimation (Platt scaling) provides no measurable benefit** over an uncalibrated risk score for this surrogate and feature set (null result, reported honestly).
> 3. Risk features (dominated by the normalized PDE residual) predict surrogate failure well (AUROC ≈ 0.99 in-distribution), but degrade under out-of-distribution viscosity shifts (AUROC ≈ 0.90, failure rate ~0.67).
> 4. Selective numerical correction substantially improves long-horizon reliability over pure neural rollout (large effect sizes), but does not reach pure-numerical accuracy at low intervention budgets.

---

## 3. What works / what does not work

| Element | Verdict | Evidence |
|---|---|---|
| Risk features predict failure | Works (in-distribution) | AUROC 0.988 ± 0.002 |
| Calibration (Platt scaling) | Does NOT work | ECE/Brier/AUROC identical; p = 0.29 vs fixed threshold |
| Budget-constrained risk-ranked intervention (G) | Works — enforces budget exactly | Actual frac = nominal at all budgets |
| Fixed-threshold / risk-threshold (C, E, F) | Overshoot budget | Nominal 0.05 → actual ~0.20–0.22 (aggregate 0.66) |
| Selective correction improves reliability | Works vs pure neural | G vs A: d = −1.45 to −2.03, p < 1e-30 |
| Reaches pure-numerical accuracy | Does NOT at low budget | G(0.2) rel L2 = 0.159 vs oracle 0.004 |
| OOD robustness | Degrades | Fail rate ~0.67 (vs ~0.09 in-dist); AUROC ~0.90 |
| Lightweight MLP (~17K params) | Adequate for the task | Beats identity baseline on next-step prediction |

---

## 4. What has NOT been demonstrated

- Distribution-free calibration guarantees under shift (exchangeability assumed; explicitly not claimed).
- Superiority over hybrid solvers such as ANCHOR/INC/HIN-LRI (different setting: lightweight MLP + budget, not neural operator).
- Benefit of calibration over raw risk.
- Benefit at matched *actual* computational cost (E/F are dominated by C on cost; G enforces budget but is less accurate than C at matched actual frac because C overshoots its nominal budget).
- Generalization beyond 1D Burgers' equation (advection-diffusion is secondary, not fully reported).

---

## 5. Recommended framing for the paper

Position BRACE-CPU as:

> *"A reproducible empirical study of risk-informed, budget-constrained selective numerical correction for lightweight neural PDE surrogates on CPU-constrained hardware. We show that (i) budget-constrained risk-ranked intervention enforces computational budgets exactly where threshold policies fail, (ii) calibrated failure-probability estimation adds no benefit over raw risk (a null result), and (iii) selective correction improves long-horizon reliability over pure neural rollout but degrades under out-of-distribution parameter shifts."*

Avoid:
- "First," "novel architecture," "state-of-the-art," "guaranteed," "optimal," "universally robust."
- Claiming calibration is a successful contribution.
- Claiming BRACE-CPU beats ANCHOR or other hybrid solvers (different setting; not compared).

---

## 6. Suitable venues (recommendation only, no acceptance claim)

Given the scope (single PDE, lightweight MLP, CPU-only, empirical null result, honest limitations), suitable venues would be those welcoming empirical scientific-ML studies with negative results, e.g., workshops track at NeurIPS/ICML/ICLR on scientific ML, or a computational physics journal emphasizing reproducible empirical evaluation. **No acceptance is claimed or implied; submission requires explicit authorization.**

*End of research positioning.*
