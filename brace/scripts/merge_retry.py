"""Merge a retrained ensemble member into an ensemble checkpoint manifest.

Used when a member diverges during training (validation rollout blow-up).
The original member's validation metrics are retained in
results/processed/training_instability.json for honest reporting.
"""
from __future__ import annotations

import json
import os
import shutil

TARGET_TAG = "burgers1d_fno1d_s1"
RETRY_TAG = "burgers1d_fno1d_s1retry"
MEMBER = 0


def main() -> None:
    with open(f"results/raw/train_{TARGET_TAG}.json") as f:
        target = json.load(f)
    with open(f"results/raw/train_{RETRY_TAG}.json") as f:
        retry = json.load(f)
    old = target["members"][MEMBER]
    new = retry["members"][0]
    record = {"target": TARGET_TAG, "member": MEMBER,
              "reason": "validation rollout divergence (relL2 > 2x ensemble median)",
              "original": old, "replacement": new}
    os.makedirs("results/processed", exist_ok=True)
    path = "results/processed/training_instability.json"
    hist = json.load(open(path)) if os.path.exists(path) else []
    hist.append(record)
    with open(path, "w") as f:
        json.dump(hist, f, indent=2)
    shutil.copyfile(f"checkpoints/{RETRY_TAG}_m0.pt", f"checkpoints/{TARGET_TAG}_m{MEMBER}.pt")
    target["members"][MEMBER] = new
    target["members"][MEMBER]["replaced_member"] = True
    with open(f"results/raw/train_{TARGET_TAG}.json", "w") as f:
        json.dump(target, f, indent=2)
    print(f"replaced {TARGET_TAG} m{MEMBER}: {old['val_rel_l2']:.4f} -> {new['val_rel_l2']:.4f}")


if __name__ == "__main__":
    main()
