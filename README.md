# README.md
# BRACE-CPU

**Budgeted Risk-Adaptive Computing Engine**

Research project: Can a lightweight neural PDE surrogate achieve a better
accuracy-versus-computational-cost trade-off by using calibrated prediction-risk
estimates to selectively invoke numerical computation?

**Constraints:** Local CPU-only, 8 GiB RAM, no GPU, no cloud compute.

## Structure
- `docs/` — environment report, literature review, novelty analysis, research protocol, status logs
- `src/` — models, solvers, uncertainty, calibration, controller, correction, evaluation, monitoring
- `experiments/` — pilot, baselines, main, ablations, robustness
- `data/` — generation, processed, metadata
- `results/` — raw, processed, figures, tables
- `checkpoints/`, `paper/`, `tests/`, `scripts/`, `configs/`

## Environment
- Isolated venv at `.venv` (Python 3.14.4)
- CPU-only PyTorch (`torch 2.15.0+cpu`)
- NumPy, SciPy, Matplotlib

## Run solver tests
```
PYTHONPATH=src .venv/bin/python -m pytest tests/ -v
```

## License
Private research.
