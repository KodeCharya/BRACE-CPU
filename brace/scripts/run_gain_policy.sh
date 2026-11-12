#!/usr/bin/env bash
# Merge the gain-aware allocator and its oracle into existing result files.
# Every call refits the risk/gain models on validation and then merges the two
# new policies, preserving all previously recorded policies.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
PY=../.venv/bin/python
T=4

run () {  # benchmark split seed ntest model tag downsample budgets...
  local bench=$1 split=$2 seed=$3 ntest=$4 model=$5 tag=$6 downsample=$7; shift 7
  local out="results/raw/main/${bench}_${split}_s${seed}_step.json"
  if [ ! -f "$out" ]; then echo "[skip: no base results] $out"; return; fi
  echo "== gain policies: $bench $split s$seed B=$* =="
  local extra=()
  if [ "$downsample" != "0" ]; then extra+=(--downsample-to "$downsample"); fi
  $PY scripts/run_experiments.py --benchmark "$bench" --split "$split" --seed "$seed" \
      --model "$model" --tag "$tag" --n-test "$ntest" --budgets "$@" \
      --policies gain_ranked oracle_gain --exp-id main --threads $T \
      --merge-records "${extra[@]}"
}

# Burger primary benchmark (two seeds for test, OOD splits as available).
run burgers1d test      0 40 fno1d burgers1d_fno1d_s0 0 0.05 0.1 0.2 0.4
run burgers1d test      1 40 fno1d burgers1d_fno1d_s1 0 0.05 0.1 0.2 0.4
run burgers1d ood_param 0 28 fno1d burgers1d_fno1d_s0 0 0.1 0.2 0.4
run burgers1d ood_param 1 28 fno1d burgers1d_fno1d_s1 0 0.1 0.2 0.4
run burgers1d ood_ic    0 28 fno1d burgers1d_fno1d_s0 0 0.1 0.2 0.4

# Secondary benchmarks.
run allencahn1d test      0 40 fno1d allencahn1d_fno1d_s0 0  0.05 0.1 0.2 0.4
run allencahn1d ood_param 0 28 fno1d allencahn1d_fno1d_s0 0  0.1 0.2 0.4
run allencahn1d ood_ic    0 28 fno1d allencahn1d_fno1d_s0 0  0.1 0.2 0.4
run advection2d test      0 40 fno2d advection2d_fno2d_s0 32 0.05 0.1 0.2 0.4
run advection2d ood_param 0 28 fno2d advection2d_fno2d_s0 32 0.1 0.2 0.4
run advection2d ood_ic    0 28 fno2d advection2d_fno2d_s0 32 0.1 0.2 0.4

echo DONE > results/raw/exp_gain_done.txt
