"""Generate all BRACE datasets. Run from the `brace/` directory:

    PYTHONPATH=. ../.venv/bin/python scripts/generate_data.py --benchmark burgers1d
"""
from __future__ import annotations

import argparse
import os

from src.data.generate import BENCHMARKS, SPLITS, generate_split


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", choices=BENCHMARKS, required=True)
    ap.add_argument("--splits", nargs="+", default=list(SPLITS))
    ap.add_argument("--base-seed", type=int, default=20261111)
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--n-traj", type=int, default=None, help="override size (debug only)")
    args = ap.parse_args()
    for split in args.splits:
        print(f"== {args.benchmark} / {split} ==", flush=True)
        meta = generate_split(args.benchmark, split, base_seed=args.base_seed,
                              out_dir=args.out_dir, n_traj=args.n_traj)
        print(f"   -> {meta['sha256'][:16]} shape={meta['shape']}", flush=True)


if __name__ == "__main__":
    main()
