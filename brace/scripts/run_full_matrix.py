"""Sequential orchestrator for the full BRACE experiment matrix.

Skips any run whose output JSON already exists (resumable). Every command is
logged to results/raw/orchestrator.log.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time

PY = "../.venv/bin/python"


def run(cmd: list[str], log_path: str, skip_if: str | None = None) -> bool:
    if skip_if and os.path.exists(skip_if):
        print(f"[skip] {skip_if}", flush=True)
        return False
    t0 = time.time()
    with open(log_path, "a") as log:
        log.write("\n$ " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    status = "ok" if proc.returncode == 0 else f"FAILED({proc.returncode})"
    print(f"[{status}] {time.time() - t0:.0f}s :: {' '.join(cmd[:6])}", flush=True)
    return proc.returncode == 0


def exp(args: list[str], exp_id: str, benchmark: str, split: str, seed: int,
        mode: str = "step", tag: str | None = None) -> None:
    suffix = mode
    out = f"results/raw/{exp_id}/{benchmark}_{split}_s{seed}_{suffix}.json"
    cmd = [PY, "scripts/run_experiments.py", "--benchmark", benchmark,
           "--split", split, "--seed", str(seed), "--exp-id", exp_id,
           "--correction-mode", mode, *args]
    if tag:
        cmd += ["--tag", tag]
    run(cmd, "results/raw/orchestrator.log", skip_if=out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all",
                    choices=["burgers", "secondary", "ablations", "correction", "all"])
    args = ap.parse_args()

    if args.phase in ("burgers", "all"):
        for seed in (0, 1):
            for split, budgets in [("test", ["0.05", "0.1", "0.2", "0.4"]),
                                   ("ood_param", ["0.1", "0.2", "0.4"]),
                                   ("ood_ic", ["0.1", "0.2", "0.4"])]:
                exp(["--budgets", *budgets, "--n-test", "40"], "main",
                    "burgers1d", split, seed)

    if args.phase in ("secondary", "all"):
        # Allen-Cahn
        exp(["--budgets", "0.1", "0.2", "0.4", "--n-test", "36"], "main",
            "allencahn1d", "test", 0)
        for split in ("ood_param", "ood_ic"):
            exp(["--budgets", "0.1", "0.2", "0.4", "--n-test", "36"], "main",
                "allencahn1d", split, 0)
        # 2D advection-diffusion (trained/evaluated on a 32x32 grid)
        exp(["--budgets", "0.1", "0.2", "0.4", "--n-test", "24", "--downsample-to", "32"],
            "main", "advection2d", "test", 0)
        for split in ("ood_param", "ood_ic"):
            exp(["--budgets", "0.1", "0.2", "0.4", "--n-test", "24", "--downsample-to", "32"],
                "main", "advection2d", split, 0)

    if args.phase in ("ablations", "all"):
        base = ["--budgets", "0.2", "0.4", "--n-test", "40", "--seed", "0",
                "--policies", "neural", "brace"]
        variants = {
            "no_residual": ["--ablate", "no_residual"],
            "no_temporal": ["--ablate", "no_temporal"],
            "no_ensemble_feat": ["--ablate", "no_ensemble_feat"],
            "residual_only": ["--ablate", "residual_only"],
            "single_member": ["--n-members", "1"],
            "cal_frac025": ["--cal-frac", "0.25"],
            "brace_prob": ["--brace-mode", "prob"],
            "brace_expected": ["--brace-mode", "expected"],
            "time_weight_early": ["--time-weight", "early"],
            "val20": ["--n-val", "20"],
            "val40": ["--n-val", "40"],
        }
        for name, extra in variants.items():
            out = f"results/raw/ablation/burgers1d_test_s0_step_{name}.json"
            if os.path.exists(out):
                continue
            cmd = [PY, "scripts/run_experiments.py", "--benchmark", "burgers1d",
                   "--split", "test", "--exp-id", "ablation", "--variant", name, *base, *extra]
            run(cmd, "results/raw/orchestrator.log", skip_if=out)

    if args.phase in ("correction", "all"):
        for mode, extra in [("window", ["--window", "5"]),
                            ("window", ["--window", "10"])]:
            w = extra[1]
            out = f"results/raw/correction/burgers1d_test_s0_window_w{w}.json"
            if os.path.exists(out):
                continue
            cmd = [PY, "scripts/run_experiments.py", "--benchmark", "burgers1d",
                   "--split", "test", "--seed", "0", "--exp-id", "correction",
                   "--budgets", "0.1", "0.2", "0.4", "--n-test", "40",
                   "--policies", "neural", "ranked_residual", "brace",
                   "--correction-mode", mode, *extra]
            run(cmd, "results/raw/orchestrator.log", skip_if=out)


if __name__ == "__main__":
    main()
