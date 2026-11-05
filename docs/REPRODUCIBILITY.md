# Reproducibility — BRACE-CPU

**Project:** BRACE-CPU — Budgeted Risk-Adaptive Computing Engine  
**Date:** 2026-11-05

## 1. Environment

- OS: Ubuntu 26.04.1 LTS, Linux 7.0.0-42-generic x86_64
- CPU: Intel Core i5-6500 @ 3.20 GHz, 4 cores / 4 threads, 1 socket
- RAM: 7.2 GiB total (~5.3 GiB available at audit)
- GPU: none; CUDA unavailable
- Disk: 112 GiB (/home), 88 GiB available
- Python: 3.14.4 (GCC 15.2.0)
- Isolated venv: `BRACE-CPU/.venv`
- Packages (CPU-only): numpy 2.5.4, scipy 1.18.1, matplotlib 3.11.2, PyYAML, tqdm, pytest, torch 2.15.0+cpu

## 2. How to reproduce

All commands run from the project root `BRACE-CPU/` with the venv activated.

### Activate environment
```
source .venv/bin/activate
```

### Run solver tests
```
PYTHONPATH=src .venv/bin/python -m pytest tests/ -v
```

### Regenerate the dataset (Burgers, 675 trajectories)
```
PYTHONPATH=src .venv/bin/python src/data/generation/dataset.py \
  --output-dir data/processed --nx 64 --n-train 300 --n-val 75 --n-cal 75 \
  --n-test 150 --n-ood 75 --base-seed 0
```
Checksums are recorded in `data/processed/burgers/dataset_manifest.json` (SHA-256 per trajectory and per manifest).

### Train a surrogate (one seed)
```
PYTHONPATH=src .venv/bin/python - <<'EOF'
from models.surrogate import NeuralSurrogate, SurrogateConfig
from models.trainer import train_surrogate
cfg = SurrogateConfig(nx=64, n_params=1, n_time=1, hidden_layers=3, hidden_units=64,
                      activation='relu', lr=1e-3, batch_size=256, epochs=40, seed=0, device='cpu')
s = NeuralSurrogate(cfg)
train_surrogate(s, 'data/processed/burgers/train', val_dir='data/processed/burgers/validation', stride=4, device='cpu')
s.save('checkpoints/surrogate_seed0.pt')
EOF
```

### Run the full experiment matrix
```
PYTHONPATH=src .venv/bin/python scripts/run_experiment.py \
  --seeds 0 1 2 3 4 --n-test 25 --n-steps 60 --epochs 40 \
  --budgets 0.05 0.1 0.2 0.4 0.6
```
(Sequential, CPU-only; wall-clock ~ tens of minutes on this hardware.)

### Analyze results
```
PYTHONPATH=src .venv/bin/python scripts/analyze_results.py results/raw
```
Writes `results/processed/summary_by_policy.csv` and figures.

## 3. What is saved (immutable)

- `data/processed/burgers/{train,validation,calibration,test,ood_test}/*.npz + *.json` — trajectories + per-trajectory metadata + SHA-256
- `data/processed/burgers/dataset_manifest.json` — manifest + manifest SHA-256
- `checkpoints/surrogate_seed{s}.pt` — trained surrogates (FP32)
- `results/raw/<experiment_id>/` — per-record JSON results (incremental, never overwritten)
- `results/raw/<experiment_id>_all.json` — combined raw results
- `results/processed/` — aggregated CSVs + figures
- `docs/EXPERIMENT_LOG.md` — experiment log

## 4. Randomness and seeds

- Dataset: `base_seed=0`; per-trajectory seeds derived deterministically.
- Surrogate training: `torch.manual_seed(seed)`, `np.random.seed(seed)`.
- Risk estimator fitting: `seed` passed to `RiskEstimator`.
- Main experiments: seeds 0,1,2,3,4.
- Identical test trajectories used for all paired policy comparisons within a seed.

## 5. Determinism notes

- PyTorch CPU ops are deterministic for a fixed seed and thread count; default `torch.get_num_threads() = 4` (logical cores).
- NumPy RNG seeded per trajectory for dataset generation.
- No multiprocessing; single-threaded experiment execution.

## 6. Known variability

- Wall-clock times vary with CPU frequency scaling and background load; report mean/std, not single runs.
- Small-magnitude floating differences possible across runs due to thread scheduling; within CI.

## 7. Contact / artifacts

All artifacts live under `BRACE-CPU/`. No external data, credentials, or cloud resources used.
