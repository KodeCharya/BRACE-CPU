"""Train an ensemble of neural PDE surrogates.

Example:
    PYTHONPATH=. ../.venv/bin/python scripts/train_surrogates.py \
        --benchmark burgers1d --model fno1d --members 3 --seed 0 --epochs 40
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from src.models.fno import build_model
from src.models.trainer import evaluate_rollout, save_checkpoint, train_surrogate


def load_data(benchmark: str, split: str):
    z = np.load(f"data/processed/{benchmark}_{split}.npz")
    keys = ("nu", "cx", "cy", "eps")
    params = {k: z[k] for k in keys if k in z}
    return z["u"], float(z["dt_save"]), params


def downsample2d(u: np.ndarray, n_out: int = 32) -> np.ndarray:
    """Exact Fourier truncation from an n x n periodic grid to n_out x n_out."""
    n = u.shape[-1]
    if n == n_out:
        return u
    uh = np.fft.rfft2(u, axes=(-2, -1))
    uh = uh[..., :n_out, : n_out // 2 + 1]
    return np.fft.irfft2(uh, s=(n_out, n_out), axes=(-2, -1)).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--model", default="fno1d")
    ap.add_argument("--members", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--stride", type=int, default=2, help="starting-time subsampling")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--width", type=int, default=24)
    ap.add_argument("--modes", type=int, default=12)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--downsample-to", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    u_train, dt, p_train = load_data(args.benchmark, "train")
    u_val, _, _ = load_data(args.benchmark, "val")
    if args.downsample_to:
        u_train = downsample2d(u_train, args.downsample_to)
        u_val = downsample2d(u_val, args.downsample_to)

    n_points = u_train.shape[-1]
    tag = args.tag or f"{args.benchmark}_{args.model}_s{args.seed}"
    out_dir = "checkpoints"
    os.makedirs(out_dir, exist_ok=True)
    results = []
    for m in range(args.members):
        member_seed = args.seed * 1000 + m
        torch.manual_seed(member_seed)
        np.random.seed(member_seed)
        if args.model == "fno2d":
            model = build_model("fno2d", width=args.width, modes1=args.modes,
                                modes2=args.modes, layers=args.layers)
        elif args.model == "deeponet1d":
            model = build_model("deeponet1d", n_points=n_points)
        else:
            model = build_model("fno1d", width=args.width, modes=args.modes,
                                layers=args.layers)
        log_path = f"results/raw/train_{tag}_m{m}.log"
        os.makedirs("results/raw", exist_ok=True)
        open(log_path, "w").close()
        print(f"[{tag}] member {m} seed={member_seed}", flush=True)
        t0 = time.time()
        info = train_surrogate(
            model, u_train, u_val, epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, pair_stride=args.stride, seed=member_seed,
            log_path=log_path, device="cpu",
        )
        val = evaluate_rollout(model, u_val, max_traj=40)
        rec = {
            "member": m, "seed": member_seed, "val_rel_l2": val["rel_l2"],
            "val_final_rel_l2": val["final_rel_l2"], "train_seconds": info["train_seconds"],
            "epochs": args.epochs, "config": vars(args),
        }
        save_checkpoint(os.path.join(out_dir, f"{tag}_m{m}.pt"), model, rec)
        results.append(rec)
        print(f"   val relL2={val['rel_l2']:.4f} final={val['final_rel_l2']:.4f} "
              f"({info['train_seconds']:.0f}s)", flush=True)
    summary = {"tag": tag, "benchmark": args.benchmark, "model": args.model,
               "members": results, "config": vars(args), "dt_save": dt}
    with open(f"results/raw/train_{tag}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({r["member"]: round(r["val_rel_l2"], 4) for r in results}))


if __name__ == "__main__":
    main()
