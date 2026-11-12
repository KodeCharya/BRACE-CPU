#!/usr/bin/env bash
# Train the secondary benchmarks and the DeepONet baseline.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
PY=../.venv/bin/python

if [ ! -f results/raw/train_allencahn1d_fno1d_s0.json ]; then
  $PY scripts/train_surrogates.py --benchmark allencahn1d --model fno1d --members 3 \
      --seed 0 --epochs 24 --stride 2 --width 32 --modes 24 --layers 4 \
      --tag allencahn1d_fno1d_s0
fi

if [ ! -f results/raw/train_advection2d_fno2d_s0.json ]; then
  $PY scripts/train_surrogates.py --benchmark advection2d --model fno2d --members 2 \
      --seed 0 --epochs 20 --stride 5 --width 24 --modes 12 --layers 4 \
      --downsample-to 32 --tag advection2d_fno2d_s0
fi

if [ ! -f results/raw/train_burgers1d_deeponet1d_s0.json ]; then
  $PY scripts/train_surrogates.py --benchmark burgers1d --model deeponet1d --members 1 \
      --seed 0 --epochs 40 --stride 2 --tag burgers1d_deeponet1d_s0
fi
echo DONE > results/raw/train_secondary_done.txt
