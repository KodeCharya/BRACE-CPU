# Reproduction Guide — BRACE-CPU

**Date:** 2026-11-05  
**Project:** BRACE-CPU — Budgeted Risk-Adaptive Computing Engine  
**Hardware:** Intel i5-6500, 4 cores, 8 GiB RAM, CPU-only, no GPU, no cloud

---

## 1. Important reproducibility statements

- **Inference from a fixed checkpoint is deterministic and bitwise-identical** (same model + same data → same predictions).
- **Retraining is NOT bitwise-deterministic.** PyTorch CPU training exhibits modest run-to-run variation (DataLoader shuffling, CPU atomic reductions) even with `torch.manual_seed(0)`. Results are therefore reported as mean ± std / 95% CI across 5 independent seeds, which absorbs this variation.
- **Dataset generation is deterministic** given `base_seed`; per-trajectory seeds are derived deterministically; SHA-256 checksums are recorded.
- **Analysis from raw results is deterministic** (scripts recompute from raw JSON).

No external data, credentials, or cloud resources are used. Nothing is uploaded externally.

---

## 2. Environment

- OS: Ubuntu 26.04.1 LTS (Linux 7.0.0-42-generic x86_64)
- CPU: Intel Core i5-6500 @ 3.20 GHz, 4 cores / 4 threads
- RAM: 7.2 GiB total (~5.3 GiB available)
- Disk: 87 GiB available on /home
- Python: 3.14.4
- Virtual environment: `BRACE-CPU/.venv` (isolated; system Python unmodified)
- Packages (CPU-only): numpy 2.5.4, scipy 1.18.1, matplotlib 3.11.2, PyYAML, tqdm, pytest, torch 2.15.0+cpu

---

## 3. Activate the environment

All commands run from the project root `BRACE-CPU/`:

```bash
cd /home/mukesh/Documents/research/BRACE-CPU
source .venv/bin/activate
```

Verify:
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# Expected: torch 2.15.0+cpu, False
```

---

## 4. Run the test suite

```bash
PYTHONPATH=src python -m pytest tests/ -v
```

Expected: 14 passed (8 solver tests + 6 risk/controller tests). Runtime: ~2 s.

---

## 5. Regenerate the dataset (optional; already present)

The dataset already exists at `data/processed/burgers/`. To regenerate:

```bash
python src/data/generation/dataset.py \
  --output-dir data/processed \
  --nx 64 --n-train 300 --n-val 75 --n-cal 75 --n-test 150 --n-ood 75 \
  --base-seed 0
```

This produces 675 trajectories with per-trajectory metadata and SHA-256 checksums in `data/processed/burgers/dataset_manifest.json`. Runtime: several minutes on this CPU.

---

## 6. Train surrogates (one per seed)

Each seed trains a ~17K-parameter MLP. Runtime: ~4–12 min per seed on this CPU (40 epochs). Run sequentially.

```bash
python - <<'EOF'
from models.surrogate import NeuralSurrogate, SurrogateConfig
from models.trainer import train_surrogate
for seed in [0, 1, 2, 3, 4]:
    cfg = SurrogateConfig(nx=64, n_params=1, n_time=1, hidden_layers=3,
                          hidden_units=64, activation="relu", lr=1e-3,
                          batch_size=256, epochs=40, seed=seed, device="cpu")
    s = NeuralSurrogate(cfg)
    train_surrogate(s, 'data/processed/burgers/train',
                    val_dir='data/processed/burgers/validation',
                    stride=4, device='cpu')
    s.save(f'checkpoints/surrogate_seed{seed}.pt')
    print(f"seed {seed} saved")
EOF
```

Checkpoints: `checkpoints/surrogate_seed{0..4}.pt`.

**Note:** Retraining produces a somewhat different model each run (training nondeterminism). To reproduce the *reported* numbers exactly, use the existing checkpoints (do not retrain).

---

## 7. Run the full experiment matrix

```bash
python scripts/run_experiment.py \
  --seeds 0 1 2 3 4 \
  --n-test 25 --n-steps 60 \
  --epochs 40 \
  --budgets 0.05 0.1 0.2 0.4 0.6 \
  --no-retrain
```

- `--no-retrain` reuses existing checkpoints (recommended for reproducing reported numbers).
- Omit `--no-retrain` to retrain (incurs training nondeterminism).
- Runtime: sequential; on the order of tens of minutes on this CPU.
- Output: `results/raw/exp_<timestamp>_<hash>/` (per-record JSON, immutable) and `results/raw/exp_<timestamp>_<hash>_all.json` (combined).

---

## 8. Run risk/calibration evaluation (RQ2) on test set

```bash
python - <<'EOF'
import numpy as np, json
from models.surrogate import NeuralSurrogate
from models.trainer import collect_npz_paths
from uncertainty.risk_estimator import RiskEstimator
from uncertainty.calibrator import (expected_calibration_error, brier_score,
                                     failure_metrics)
from solvers.base import relative_l2_error
from uncertainty.features import extract_features

TAU = 0.01; L = 1.0
results = []
for seed in [0, 1, 2, 3, 4]:
    s = NeuralSurrogate(); s.load(f'checkpoints/surrogate_seed{seed}.pt', device='cpu')
    re = RiskEstimator(tau=TAU, L=L, seed=seed)
    re.fit(s, collect_npz_paths('data/processed/burgers/calibration'), stride=4)
    pc, pu, lb = [], [], []
    for pth in collect_npz_paths('data/processed/burgers/test'):
        d = np.load(pth); states = d['states'].astype(np.float64)
        times = d['times'].astype(np.float64); d.close()
        meta = json.load(open(pth[:-4] + '.json'))
        nu = float(meta.get('nu', 0.01)); params = np.array([nu])
        dt = float(times[1] - times[0])
        for i in range(0, states.shape[0] - 1, 4):
            u_t = states[i]; u_true = states[i+1]; t = float(times[i])
            u_pred = s.predict(u_t, params, np.array([t]))
            err = relative_l2_error(u_pred, u_true)
            feats = extract_features(s, u_t, u_pred, params, t, dt, L)
            logit = re.risk_model.predict_logit(feats[None,:])[0]
            p_unc = float(1/(1+np.exp(-logit)))
            p_cal = re.score(s, u_t, params, t, dt)
            pc.append(p_cal); pu.append(p_unc); lb.append(1.0 if err > TAU else 0.0)
    pc = np.array(pc); pu = np.array(pu); lb = np.array(lb)
    results.append(dict(seed=seed,
        cal_ece=expected_calibration_error(pc, lb), cal_brier=brier_score(pc, lb),
        cal_auroc=failure_metrics(pc, lb)['auroc'],
        unc_ece=expected_calibration_error(pu, lb), unc_brier=brier_score(pu, lb),
        unc_auroc=failure_metrics(pu, lb)['auroc']))
    print(f"seed {seed}: cal ECE={results[-1]['cal_ece']:.4f} unc ECE={results[-1]['unc_ece']:.4f}")
for k in ['cal_ece','cal_brier','cal_auroc','unc_ece','unc_brier','unc_auroc']:
    vals = [r[k] for r in results]
    print(f"{k}: mean={np.mean(vals):.4f} std={np.std(vals):.4f}")
EOF
```

Expected: calibrated ECE ≈ 0.0096, uncalibrated ≈ 0.0103; AUROC ≈ 0.988 both (calibration null result).

---

## 9. Run ablations (Phase 10)

```bash
PYTHONPATH=src python scripts/run_ablation.py > results/processed/ablation_output.txt 2>&1
```

Reuses checkpoints seed 0–2. Output: feature ablation (AUROC per feature), calibration-set-size sweep, OOD robustness (fail rate ~0.67).

---

## 10. Analyze results

```bash
PYTHONPATH=src python scripts/analyze_results.py results/raw
```

Writes `results/processed/summary_by_policy.csv` and `G_brace_budget_by_budget.csv`, plus figures.

---

## 11. LaTeX compilation

**Not possible in this environment** (no root/sudo; `pdflatex` not installed). The manuscript source `paper/main.tex` and bibliography `paper/references.bib` are syntactically valid (verified via `scripts/latex_check.py`: balanced environments, no undefined references, all citations resolve). Compile on a machine with a LaTeX toolchain before submission:

```bash
cd paper
pdflatex main.tex   # twice, for cross-references
bibtex main         # if using .bib with natbib
pdflatex main.tex
pdflatex main.tex
```

---

## 12. Expected output files

| Artifact | Path |
|---|---|
| Solver tests | `tests/test_solvers.py`, `tests/test_risk_controller.py` |
| Dataset | `data/processed/burgers/` (+ `dataset_manifest.json` with SHA-256) |
| Surrogate checkpoints | `checkpoints/surrogate_seed{0..4}.pt` |
| Raw results (authoritative) | `results/raw/exp_20261105_134910_63a5fa94_all.json` |
| Processed summary | `results/processed/summary_by_policy.csv` |
| Figures | `results/processed/figures/`, `paper/figures/` |
| Ablation output | `results/processed/ablation_output.txt` |
| Manuscript | `paper/main.tex`, `paper/references.bib` |

---

## 13. Random seeds

- Dataset: `base_seed=0`
- Surrogate training: `torch.manual_seed(seed)`, `np.random.seed(seed)` per seed (0–4)
- Risk estimator fitting: `seed` passed to `RiskEstimator`
- Main experiments: seeds 0, 1, 2, 3, 4
- Identical test trajectories used for all paired policy comparisons within a seed

---

*End of reproduction guide.*
