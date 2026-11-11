"""Dataset integrity, checksum and leakage audit."""
from __future__ import annotations

import glob
import hashlib
import json
import os

import numpy as np

BENCH = ("burgers1d", "allencahn1d", "advection2d")
SPLITS = ("train", "val", "test", "ood_param", "ood_ic")


def sha256(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main() -> None:
    report = {"checksums": {}, "shapes": {}, "stats": {}, "leakage": {}, "ood_ranges": {}}
    ic_hashes = {}
    for bench in BENCH:
        for split in SPLITS:
            npz = f"data/processed/{bench}_{split}.npz"
            meta_path = f"data/metadata/{bench}_{split}.json"
            if not os.path.exists(npz):
                continue
            meta = json.load(open(meta_path))
            digest = sha256(npz)
            report["checksums"][f"{bench}/{split}"] = {
                "sha256": digest,
                "matches_metadata": digest == meta["sha256"],
                "generation_seconds": meta["generation_seconds"],
            }
            z = np.load(npz)
            u = z["u"]
            report["shapes"][f"{bench}/{split}"] = list(u.shape)
            report["stats"][f"{bench}/{split}"] = {
                "min": float(u.min()), "max": float(u.max()),
                "mean": float(u.mean()), "std": float(u.std()),
                "all_finite": bool(np.isfinite(u).all()),
                "u0_rms_mean": float(np.sqrt((u[:, 0].reshape(u.shape[0], -1) ** 2).mean(axis=1)).mean()),
                "uT_rms_mean": float(np.sqrt((u[:, -1].reshape(u.shape[0], -1) ** 2).mean(axis=1)).mean()),
            }
            h = [hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest() for x in u[:, 0]]
            ic_hashes[(bench, split)] = set(h)
            report["ood_ranges"][f"{bench}/{split}"] = {
                k: [float(z[k].min()), float(z[k].max())]
                for k in ("nu", "cx", "cy", "eps", "length_scale", "amplitude") if k in z
            }
    # Initial-condition leakage between partitions
    for bench in BENCH:
        keys = [k for k in ic_hashes if k[0] == bench]
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                overlap = len(ic_hashes[a] & ic_hashes[b])
                report["leakage"][f"{a[1]}_vs_{b[1]}_{bench}"] = overlap
    with open("results/processed/dataset_audit.json", "w") as f:
        json.dump(report, f, indent=2)
    bad = [k for k, v in report["checksums"].items() if not v["matches_metadata"]]
    leaks = [k for k, v in report["leakage"].items() if v > 0]
    nonfinite = [k for k, v in report["stats"].items() if not v["all_finite"]]
    print(json.dumps({"n_datasets": len(report["checksums"]),
                      "checksum_mismatches": bad, "leakage_overlaps": leaks,
                      "nonfinite": nonfinite}, indent=2))


if __name__ == "__main__":
    main()
