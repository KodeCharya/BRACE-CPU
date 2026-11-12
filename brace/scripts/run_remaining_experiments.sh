#!/usr/bin/env bash
# Runs executed after the primary Burgers matrix and the surrogate training.
# Strictly sequential: concurrent heavy jobs cause severe CPU oversubscription
# on this 4-core machine.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
PY=../.venv/bin/python
T=4
mkdir -p results/raw/ablation results/raw/correction results/raw/main

# ---------------------------------------------------------------- extra main
# brace_combo was added after the seed-0/1 test matrices were produced.
for s in 0 1; do
  out="results/raw/main/burgers1d_test_s${s}_step.json"
  if ! $PY - "$out" <<'PYEOF'
import json,sys
recs=json.load(open(sys.argv[1]))["records"]
sys.exit(0 if any(r["policy"]=="brace_combo" for r in recs) else 1)
PYEOF
  then
    echo "== brace_combo on burgers test s$s =="
    $PY scripts/run_experiments.py --benchmark burgers1d --split test --seed "$s" \
        --budgets 0.05 0.1 0.2 0.4 --n-test 40 --exp-id main --threads $T \
        --policies neural uniform early brace brace_combo ranked_ensemble --merge-records
  fi
done

# ------------------------------------------------------------------ ablations
run_ablation () {  # name, extra args...
  local name=$1; shift
  local out="results/raw/ablation/burgers1d_test_s0_step_${name}.json"
  [ -f "$out" ] && { echo "[skip] $name"; return; }
  echo "== ablation: $name =="
  $PY scripts/run_experiments.py --benchmark burgers1d --split test --seed 0 \
      --budgets 0.2 0.4 --n-test 40 --exp-id ablation --variant "$name" \
      --policies neural brace --threads $T "$@"
}

run_ablation full
run_ablation no_residual        --ablate no_residual
run_ablation no_temporal        --ablate no_temporal
run_ablation no_ensemble_feat   --ablate no_ensemble_feat
run_ablation residual_only      --ablate residual_only
run_ablation single_member      --n-members 1
run_ablation myopic_expected    --brace-mode expected
run_ablation probability        --brace-mode prob
run_ablation time_weight_early  --time-weight early
run_ablation gamma_090          --gamma 0.90
run_ablation gamma_099          --gamma 0.99
run_ablation cal_frac_025       --cal-frac 0.25

# ---------------------------------------------------------------- corrections
for w in 5 10; do
  out="results/raw/correction/burgers1d_test_s0_window_w${w}.json"
  [ -f "$out" ] && { echo "[skip] window w=$w"; continue; }
  echo "== correction: window w=$w =="
  $PY scripts/run_experiments.py --benchmark burgers1d --split test --seed 0 \
      --budgets 0.1 0.2 0.4 --n-test 40 --exp-id correction \
      --policies neural ranked_residual ranked_ensemble brace brace_combo \
      --correction-mode window --window "$w" --threads $T
done

# ------------------------------------------------------------- DeepONet (1D)
out="results/raw/main/burgers1d_test_s0_step_deeponet.json"
if [ ! -f "$out" ]; then
  echo "== DeepONet baseline =="
  $PY scripts/run_experiments.py --benchmark burgers1d --split test --seed 0 \
      --model deeponet1d --tag burgers1d_deeponet1d_s0 \
      --budgets 0.1 0.2 0.4 --n-test 40 --exp-id deeponet --threads $T \
      --policies neural brace ranked_residual ranked_ensemble
fi

echo DONE > results/raw/exp_remaining_done.txt
