#!/usr/bin/env bash
# End-to-end reproduction script (CPU-only path).
# Run from the `brace/` directory.
set -euo pipefail
PY=${PY:-../.venv/bin/python}
export PYTHONPATH=.

echo "[1/7] solver tests"
$PY -m pytest tests/test_solvers.py -q

echo "[2/7] dataset generation (skips if files exist)"
for bench in burgers1d allencahn1d advection2d; do
  if [ ! -f "data/processed/${bench}_test.npz" ]; then
    $PY scripts/generate_data.py --benchmark "$bench"
  fi
done

echo "[3/7] dataset audit"
$PY scripts/audit_datasets.py

echo "[4/7] surrogate training"
if [ ! -f results/raw/train_burgers1d_fno1d_s0.json ]; then
  $PY scripts/train_surrogates.py --benchmark burgers1d --model fno1d --members 3 \
      --seed 0 --epochs 30 --stride 2 --width 32 --modes 32 --layers 4
fi
if [ ! -f results/raw/train_allencahn1d_fno1d_s0.json ]; then
  $PY scripts/train_surrogates.py --benchmark allencahn1d --model fno1d --members 3 \
      --seed 0 --epochs 24 --stride 2 --width 32 --modes 24 --layers 4
fi
if [ ! -f results/raw/train_advection2d_fno2d_s0.json ]; then
  $PY scripts/train_surrogates.py --benchmark advection2d --model fno2d --members 2 \
      --seed 0 --epochs 20 --stride 5 --width 24 --modes 12 --layers 4 --downsample-to 32
fi

echo "[5/7] main experiment matrix"
$PY scripts/run_full_matrix.py --phase all

echo "[6/7] aggregation, statistics and figures"
$PY scripts/analyze.py --exp-id main
$PY scripts/analyze.py --exp-id ablation || true
$PY scripts/analyze.py --exp-id correction || true

echo "[7/7] compute benchmark + test suite"
$PY scripts/benchmark_compute.py
$PY -m pytest tests -q
echo "done"
