#!/usr/bin/env bash
# Remaining primary-benchmark runs after seed 0 / test.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
PY=../.venv/bin/python
mkdir -p results/raw/main

run () {  # benchmark split seed ntest budgets...
  local bench=$1 split=$2 seed=$3 ntest=$4; shift 4
  local out="results/raw/main/${bench}_${split}_s${seed}_step.json"
  if [ -f "$out" ]; then echo "[skip] $out"; return; fi
  echo "[run] $bench $split s$seed n=$ntest B=$*"
  $PY scripts/run_experiments.py --benchmark "$bench" --split "$split" --seed "$seed" \
      --budgets "$@" --n-test "$ntest" --exp-id main --threads 4
}

run burgers1d test 1 40 0.05 0.1 0.2 0.4
run burgers1d ood_param 0 28 0.1 0.2 0.4
run burgers1d ood_ic 0 28 0.1 0.2 0.4
run burgers1d ood_param 1 28 0.1 0.2 0.4
echo DONE > results/raw/exp_burgers_remaining_done.txt
