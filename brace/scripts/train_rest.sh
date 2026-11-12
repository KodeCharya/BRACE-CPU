#!/usr/bin/env bash
# Sequential training of the remaining surrogates. No parallel heavy jobs:
# CPU oversubscription previously slowed FNO training by >10x.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
PY=../.venv/bin/python
T=4

if [ ! -f results/raw/train_burgers1d_deeponet1d_s0.json ]; then
  echo "== DeepONet (Burgers) =="
  $PY scripts/train_surrogates.py --benchmark burgers1d --model deeponet1d \
      --members 1 --seed 0 --epochs 40 --stride 2 --threads $T \
      --tag burgers1d_deeponet1d_s0
fi

if [ ! -f results/raw/train_allencahn1d_fno1d_s0m12.json ]; then
  echo "== Allen-Cahn members 1-2 =="
  $PY scripts/train_surrogates.py --benchmark allencahn1d --model fno1d \
      --members 2 --seed 1 --epochs 16 --stride 2 --width 32 --modes 24 --layers 4 \
      --threads $T --tag allencahn1d_fno1d_s0m12
fi

if [ ! -f results/raw/train_advection2d_fno2d_s0.json ]; then
  echo "== Advection-diffusion 2D =="
  $PY scripts/train_surrogates.py --benchmark advection2d --model fno2d \
      --members 2 --seed 0 --epochs 12 --stride 10 --width 16 --modes 8 --layers 3 \
      --downsample-to 32 --threads $T --tag advection2d_fno2d_s0
fi
echo DONE > results/raw/train_rest_done.txt
