"""Diagnose one-step accuracy versus autoregressive rollout stability.

A surrogate can have a small one-step error and still drift badly once its own
predictions are fed back. This is reported separately for every trained
ensemble member so the comparison between architectures is not confounded by
training set size.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from src.models.fno import build_model
from src.models.trainer import evaluate_rollout


def load_split(benchmark: str, split: str, downsample_to: int = 0):
    z = np.load(f"data/processed/{benchmark}_{split}.npz")
    u = z["u"]
    if downsample_to:
        uh = np.fft.rfft2(u, axes=(-2, -1))[..., :downsample_to, : downsample_to // 2 + 1]
        u = np.fft.irfft2(uh, s=(downsample_to, downsample_to), axes=(-2, -1)).astype(np.float32)
    return u.astype(np.float32)


def build_from_meta(meta: dict):
    cfg = meta["config"]
    if meta["model"] == "fno2d":
        return build_model("fno2d", width=cfg["width"], modes1=cfg["modes"],
                           modes2=cfg["modes"], layers=cfg["layers"])
    if meta["model"] == "deeponet1d":
        return build_model("deeponet1d", n_points=128)
    return build_model("fno1d", width=cfg["width"], modes=cfg["modes"], layers=cfg["layers"])


@torch.no_grad()
def one_step_error(model, u: np.ndarray, max_traj: int = 40) -> dict:
    uu = torch.as_tensor(u[:max_traj], dtype=torch.float32)
    model.eval()
    errs = []
    for t in range(uu.shape[1] - 1):
        pred = model(uu[:, t])
        target = uu[:, t + 1]
        num = ((pred - target) ** 2).sum(dim=tuple(range(1, pred.dim())))
        den = (target ** 2).sum(dim=tuple(range(1, target.dim()))) + 1e-12
        errs.append(torch.sqrt(num / den))
    e = torch.stack(errs, dim=1).numpy()
    return {"one_step_mean": float(e.mean()), "one_step_final": float(e[:, -1].mean()),
            "one_step_max": float(e.max()), "per_step_mean": e.mean(axis=0).tolist()}


def diagnose(tag: str, benchmark: str, split: str = "val", downsample_to: int = 0,
             max_traj: int = 40) -> list[dict]:
    meta = json.load(open(f"results/raw/train_{tag}.json"))
    u = load_split(benchmark, split, downsample_to)
    rows = []
    for rec in meta["members"]:
        model = build_from_meta(meta)
        ck = torch.load(f"checkpoints/{tag}_m{rec['member']}.pt", map_location="cpu",
                        weights_only=False)
        model.load_state_dict(ck["state_dict"])
        model.eval()
        one = one_step_error(model, u, max_traj)
        roll = evaluate_rollout(model, u, max_traj=max_traj)
        growth = float(roll["per_step"][:, -1].mean() / (one["one_step_mean"] + 1e-12))
        rows.append({"tag": tag, "benchmark": benchmark, "split": split,
                     "member": rec["member"], "val_rel_l2_manifest": rec["val_rel_l2"],
                     **one, "rollout_mean": roll["rel_l2"],
                     "rollout_final": roll["final_rel_l2"],
                     "rollout_growth_ratio": growth,
                     "rollout_per_step_mean": roll["per_step"].mean(axis=0).tolist()})
        print(f"{tag} m{rec['member']}: one-step {one['one_step_mean']:.4f} "
              f"rollout {roll['rel_l2']:.4f} growth {growth:.1f}x", flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/processed/baseline_diagnostics.json")
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    jobs = [
        ("burgers1d_fno1d_s0", "burgers1d", 0),
        ("burgers1d_deeponet1d_s0", "burgers1d", 0),
        ("allencahn1d_fno1d_s0", "allencahn1d", 0),
        ("advection2d_fno2d_s0", "advection2d", 32),
    ]
    rows = []
    for tag, bench, ds in jobs:
        if not os.path.exists(f"results/raw/train_{tag}.json"):
            print(f"[skip] {tag}: no manifest")
            continue
        rows.extend(diagnose(tag, bench, "val", ds))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"diagnostics": rows}, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
