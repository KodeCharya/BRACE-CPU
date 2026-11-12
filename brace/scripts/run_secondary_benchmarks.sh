#!/usr/bin/env bash
# Evaluation matrix for the two secondary benchmarks (Allen-Cahn 1D,
# advection-diffusion 2D). Run only after their ensembles are trained and only
# when the machine is otherwise idle: every heavy job here is sequential.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
PY=../.venv/bin/python
T=4
POL="neural numerical fixed adaptive uniform early ranked_residual ranked_ensemble ranked_calibrated brace brace_combo oracle reanchor"
mkdir -p results/raw/main

run () {  # benchmark split ntest model tag downsample budgets...
  local bench=$1 split=$2 ntest=$3 model=$4 tag=$5 downsample=$6; shift 6
  local out="results/raw/main/${bench}_${split}_s0_step.json"
  if [ -f "$out" ]; then echo "[skip] $out"; return; fi
  echo "== $bench $split B=$* =="
  local extra=()
  if [ "$downsample" != "0" ]; then extra+=(--downsample-to "$downsample"); fi
  $PY scripts/run_experiments.py --benchmark "$bench" --split "$split" --seed 0 \
      --model "$model" --tag "$tag" --n-test "$ntest" --budgets "$@" \
      --policies $POL --exp-id main --threads $T "${extra[@]}"
}

# Allen-Cahn: FNO-1D, three members (one 24-epoch member, two 16-epoch members).
run allencahn1d test      40 fno1d allencahn1d_fno1d_s0 0  0.05 0.1 0.2 0.4
run allencahn1d ood_param 28 fno1d allencahn1d_fno1d_s0 0  0.1 0.2 0.4
run allencahn1d ood_ic    28 fno1d allencahn1d_fno1d_s0 0  0.1 0.2 0.4

# Advection-diffusion: FNO-2D on a 32x32 Fourier truncation. Corrections use the
# same 32x32 grid, which reproduces the truncated reference data exactly.
run advection2d test      40 fno2d advection2d_fno2d_s0 32 0.05 0.1 0.2 0.4
run advection2d ood_param 28 fno2d advection2d_fno2d_s0 32 0.1 0.2 0.4
run advection2d ood_ic    28 fno2d advection2d_fno2d_s0 32 0.1 0.2 0.4

echo DONE > results/raw/exp_secondary_done.txt
