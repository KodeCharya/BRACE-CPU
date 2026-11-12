"""Append ensemble members trained under a temporary tag into a target manifest.

Members are sometimes trained in separate processes (e.g. after an interrupted
run). This script copies the checkpoints to the target tag and appends their
records to the target training manifest, preserving each member's own config.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-tag", required=True)
    ap.add_argument("--source-tag", required=True)
    ap.add_argument("--source-members", nargs="+", type=int, required=True)
    ap.add_argument("--dest-indices", nargs="+", type=int, required=True)
    args = ap.parse_args()
    if len(args.source_members) != len(args.dest_indices):
        raise SystemExit("source-members and dest-indices must have equal length")

    target_path = f"results/raw/train_{args.target_tag}.json"
    source_path = f"results/raw/train_{args.source_tag}.json"
    with open(target_path) as f:
        target = json.load(f)
    with open(source_path) as f:
        source = json.load(f)

    for src, dst in zip(args.source_members, args.dest_indices):
        rec = dict(source["members"][src])
        rec["member"] = dst
        rec["merged_from"] = {"tag": args.source_tag, "member": src}
        shutil.copyfile(f"checkpoints/{args.source_tag}_m{src}.pt",
                        f"checkpoints/{args.target_tag}_m{dst}.pt")
        target["members"].append(rec)

    target["members"].sort(key=lambda r: r["member"])
    target["merged_members"] = target.get("merged_members", []) + [args.source_tag]
    with open(target_path, "w") as f:
        json.dump(target, f, indent=2)
    vals = [round(m["val_rel_l2"], 5) for m in target["members"]]
    print(f"{args.target_tag}: {len(target['members'])} members, val relL2 {vals}")


if __name__ == "__main__":
    main()
