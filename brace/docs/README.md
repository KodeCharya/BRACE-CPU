# BRACE — Budgeted Risk-Adaptive Correction Engine

BRACE is the code, data pipeline and experiment harness for a study of
**budgeted numerical correction of neural PDE surrogates**. Given a trained
surrogate ensemble, a fixed budget measured as a fraction of the cost of running
the reference solver for the whole horizon, BRACE decides *at which steps* to
replace the surrogate prediction with one saved interval of the reference
solver.

The repository contains the full experiment matrix behind the accompanying
manuscript draft in `paper/`. All tables and figures in the manuscript are
generated from `results/raw/` by the scripts in `scripts/`; prose numbers are
transcribed from the same aggregated CSVs. No result is entered by hand.

## Headline result

The allocation *heuristic* matters much less than *where in the horizon* the
budget is spent. A learned, calibrated horizon-risk model does not beat a
training-free ensemble-disagreement ranking; a learned counterfactual-gain model
is worse than both; and even a counterfactual-gain **oracle** loses to a trivial
"correct the first *k* steps" schedule at large budgets. The reason is
measurable: the relative value of a correction decays along the rollout horizon,
and under distribution shift it becomes *anti-correlated* with the surrogate's
own error (Spearman −0.84 on shifted Burgers), so ranking steps by risk targets
the steps where correcting helps least.

## Layout

```
brace/
  src/
    solvers/        reference PDE solvers (Fourier pseudo-spectral)
    data/           trajectory generation, parameter draws, split definitions
    models/         FNO-1D/2D, DeepONet, trainer, rollout evaluation
    uncertainty/    online risk features, risk model, gain model, calibration
    calibration/    split conformal prediction
    controller/     policies (top-k, adaptive), closed-loop execution, diagnostics
    correction/     numerical correction engine and exact cost accounting
    evaluation/     paired statistics (bootstrap, t intervals, effect size)
  scripts/          data generation, training, experiment matrices, analysis
  tests/            unit tests (solvers, calibration, policies, statistics)
  data/processed/   generated datasets (.npz) + metadata
  checkpoints/      trained ensemble members (.pt)
  results/raw/      unmodified JSON records written by experiment runs
  results/tables/   aggregated CSVs and figures
  results/processed/ summary JSON, results digest, diagnostics
  paper/            manuscript sources and generated LaTeX tables
```

## Environment

CPU only: 4 cores, 7.3 GiB RAM, Python 3.14, `torch 2.15.0+cpu`,
`numpy 2.5.4`, `scipy 1.18.1`, `matplotlib 3.11.2`, `pytest 9.2.0`.
No GPU timings are reported anywhere; all wall-clock numbers are CPU numbers
measured on an idle machine.

```bash
python -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install numpy scipy matplotlib pytest
```

## Reproducing

All commands run from `brace/` with `PYTHONPATH=.` and the venv interpreter.
Heavy jobs must run **sequentially**: concurrent 4-thread jobs oversubscribe the
4 available cores and slow each other by more than 10x.

```bash
export PYTHONPATH=.
PY=../.venv/bin/python

# 1. datasets (15 files: 3 benchmarks x 5 splits)
$PY scripts/generate_data.py --benchmark burgers1d
$PY scripts/generate_data.py --benchmark allencahn1d
$PY scripts/generate_data.py --benchmark advection2d
$PY scripts/audit_datasets.py              # checksums, leakage, finiteness

# 2. ensembles
$PY scripts/train_surrogates.py --benchmark burgers1d  --model fno1d      --members 3 --seed 0 --epochs 60 --tag burgers1d_fno1d_s0
$PY scripts/train_surrogates.py --benchmark burgers1d  --model fno1d      --members 3 --seed 1 --epochs 60 --tag burgers1d_fno1d_s1
$PY scripts/train_surrogates.py --benchmark burgers1d  --model deeponet1d --members 1 --seed 0 --epochs 150 --stride 2 --tag burgers1d_deeponet1d_s0
$PY scripts/train_surrogates.py --benchmark allencahn1d --model fno1d     --members 3 --seed 0 --epochs 24 --tag allencahn1d_fno1d_s0
$PY scripts/train_surrogates.py --benchmark advection2d --model fno2d     --members 2 --seed 0 --epochs 12 --downsample-to 32 --tag advection2d_fno2d_s0

# 3. experiment matrices (sequential shell drivers)
bash scripts/run_burgers_remaining.sh
bash scripts/run_remaining_experiments.sh      # ablations, windows, DeepONet, combo
bash scripts/run_secondary_benchmarks.sh       # Allen-Cahn + advection-diffusion
bash scripts/run_gain_policy.sh                # gain-ranked + gain oracle

# 4. analysis
$PY scripts/analyze.py --exp-id main
$PY scripts/analyze.py --exp-id ablation
$PY scripts/analyze.py --exp-id correction
$PY scripts/analyze.py --exp-id deeponet
$PY scripts/analyze_gain_decay.py              # mechanism diagnostic
$PY scripts/diagnose_baseline.py               # one-step vs rollout
$PY scripts/benchmark_compute.py               # latency / memory
$PY scripts/make_paper_assets.py               # LaTeX tables
$PY scripts/collect_results.py                 # auditable digest

# 5. tests
$PY -m pytest tests -q
```

`scripts/benchmark_compute.py` and `scripts/diagnose_baseline.py` must run on an
idle machine; their timing numbers are meaningless under CPU contention.

## Cost model

One correction costs exactly one *saved interval* of the reference solver. A
budget `B` therefore permits `floor(B * T)` step corrections over a horizon of
`T` saved intervals, and `B = 1` equals running the reference solver for the
whole horizon. Every ranked policy allocates exactly `floor(B * T)` actions per
trajectory, so its realised cost equals the declared budget (verified: deviation
below 0.001 across all runs). Window corrections cost `W` intervals and are
constrained to start at `t >= W-1` so no window is truncated; re-anchoring at
step `t` costs `t+1` intervals.

Planning overhead is reported separately and is *not* charged to the budget:
budget accounting covers reference-solver work only. This is stated explicitly
because it favours BRACE; see `docs/results.md`.

## Data splits

| split | purpose |
|---|---|
| `train` | fitting surrogate parameters |
| `val` | fitting risk/gain models, conformal calibration, hyper-parameters |
| `test` | in-distribution evaluation |
| `ood_param` | parameter shift (viscosity / transport speed outside training range) |
| `ood_ic` | initial-condition shift (different length scale and amplitude) |

Each dataset is written with a SHA-256 checksum and split metadata;
`scripts/audit_datasets.py` verifies 15 datasets, 0 checksum mismatches, 0
initial-condition leakage between splits, 0 non-finite fields.

## Status and scope

- Burgers-1D: 2 surrogate seeds (test), 2 seeds (ood_param), 1 seed (ood_ic).
- Allen–Cahn-1D: 1 seed, 3 ensemble members.
- Advection–diffusion-2D: 1 seed, 2 ensemble members, evaluated on a 32×32
  Fourier truncation.
- DeepONet: 1 seed, 1 member (exploratory; see limitations).

The small number of seeds is a real limitation, not a presentation choice.
Seed-level intervals use Student-t multipliers and are wide with n = 2; the
single-seed benchmarks have no seed-level interval at all and are reported as
point estimates with per-trajectory paired bootstrap where applicable.
