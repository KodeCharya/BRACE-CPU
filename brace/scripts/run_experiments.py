"""Run the main policy/budget experiment matrix for one benchmark and seed.

Example:
    PYTHONPATH=. ../.venv/bin/python scripts/run_experiments.py \
        --benchmark burgers1d --seed 0 --split test --budgets 0.1 0.2 0.4
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from src.controller.policies import quantile_threshold, threshold_actions, topk_actions
from src.controller.rollout import (adaptive_execute, counterfactual_gains, execute_actions,
                                    neural_pass, plan_scores)
from src.models.fno import build_model
from src.uncertainty.pipeline import (evaluate_risk_on_split, fit_gain_model, fit_risk_model,
                                      residual_solver)
from src.uncertainty.features import FEATURE_NAMES

POLICIES = ["neural", "numerical", "fixed", "adaptive", "uniform", "early",
            "ranked_residual", "ranked_ensemble", "ranked_calibrated", "brace",
            "brace_combo", "gain_ranked", "oracle_gain", "oracle", "reanchor"]

SCORE_KIND = {
    "ranked_residual": "residual",
    "ranked_ensemble": "ensemble",
    "ranked_calibrated": "calibrated",
    "brace": "horizon",
    "brace_combo": "combo",
    "oracle": "oracle",
    "gain_ranked": "gain",
    "oracle_gain": "oracle_gain",
}


def load_split(benchmark: str, split: str, downsample_to: int = 0):
    z = np.load(f"data/processed/{benchmark}_{split}.npz")
    u = z["u"]
    if downsample_to:
        n = u.shape[-1]
        uh = np.fft.rfft2(u, axes=(-2, -1))[..., :downsample_to, : downsample_to // 2 + 1]
        u = np.fft.irfft2(uh, s=(downsample_to, downsample_to), axes=(-2, -1)).astype(np.float32)
    params = {k: z[k] for k in ("nu", "cx", "cy", "eps") if k in z}
    return u.astype(np.float32), float(z["dt_save"]), params


def load_ensemble(benchmark: str, seed: int, model_name: str = "fno1d", tag: str | None = None,
                  n_members: int | None = None):
    tag = tag or f"{benchmark}_{model_name}_s{seed}"
    with open(f"results/raw/train_{tag}.json") as f:
        meta = json.load(f)
    cfg = meta["config"]
    models = []
    n_use = len(meta["members"]) if n_members is None else min(n_members, len(meta["members"]))
    for m in range(n_use):
        if model_name == "fno2d":
            model = build_model("fno2d", width=cfg["width"], modes1=cfg["modes"],
                                modes2=cfg["modes"], layers=cfg["layers"])
        elif model_name == "deeponet1d":
            model = build_model("deeponet1d", n_points=128)
        else:
            model = build_model("fno1d", width=cfg["width"], modes=cfg["modes"],
                                layers=cfg["layers"])
        ck = torch.load(f"checkpoints/{tag}_m{m}.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ck["state_dict"])
        model.eval()
        models.append(model)
    return models, meta


def _record(policy, budget, res, extra=None):
    rec = {"policy": policy, "budget": budget, "variant": ""}
    rec.update(res["metrics"])
    rec["wall_seconds"] = res["wall_seconds"]
    rec["intervention_fraction"] = res["intervention_fraction"]
    if extra:
        rec.update(extra)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="test")
    ap.add_argument("--budgets", nargs="+", type=float, default=[0.1, 0.2, 0.4])
    ap.add_argument("--policies", nargs="+", default=POLICIES)
    ap.add_argument("--n-test", type=int, default=40)
    ap.add_argument("--model", default="fno1d")
    ap.add_argument("--tag", default=None, help="checkpoint tag override")
    ap.add_argument("--n-members", type=int, default=None)
    ap.add_argument("--ablate", default="none",
                    choices=["none", "no_residual", "no_temporal", "no_ensemble_feat", "residual_only"])
    ap.add_argument("--cal-frac", type=float, default=0.5)
    ap.add_argument("--brace-mode", default="hybrid", choices=["hybrid", "expected", "prob"])
    ap.add_argument("--time-weight", default="none", choices=["none", "early", "late"])
    ap.add_argument("--risk-target", default="step", choices=["step", "horizon"])
    ap.add_argument("--gamma", type=float, default=0.95, help="discount for horizon risk")
    ap.add_argument("--merge-records", action="store_true",
                    help="merge with an existing result file, replacing matching (policy,budget,variant)")
    ap.add_argument("--variant", default="", help="ablation label stored with each record")
    ap.add_argument("--downsample-to", type=int, default=0)
    ap.add_argument("--tau-fail", type=float, default=0.05)
    ap.add_argument("--correction-mode", default="step")
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--exp-id", default="main")
    ap.add_argument("--n-val", type=int, default=60)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    out_dir = f"results/raw/{args.exp_id}"
    os.makedirs(out_dir, exist_ok=True)

    models, meta = load_ensemble(args.benchmark, args.seed, args.model, args.tag, args.n_members)
    u_val, dt, p_val = load_split(args.benchmark, "val", args.downsample_to)
    u_test, _, p_test = load_split(args.benchmark, args.split, args.downsample_to)
    n = min(args.n_test, u_test.shape[0])
    u_val, p_val = u_val[: args.n_val], {k: v[: args.n_val] for k, v in p_val.items()}
    u_test, p_test = u_test[:n], {k: v[:n] for k, v in p_test.items()}
    T = u_test.shape[1] - 1
    solver = residual_solver(args.benchmark, dt, n=u_test.shape[-1])

    t0 = time.time()
    mask = None
    if args.ablate == "no_residual":
        mask = np.ones(9, dtype=bool); mask[2] = False
    elif args.ablate == "no_temporal":
        mask = np.ones(9, dtype=bool); mask[[5, 6, 7]] = False
    elif args.ablate == "no_ensemble_feat":
        mask = np.ones(9, dtype=bool); mask[[0, 1]] = False
    elif args.ablate == "residual_only":
        mask = np.zeros(9, dtype=bool); mask[2] = True
    risk = fit_risk_model(models, u_val, args.benchmark, p_val, dt, tau_fail=args.tau_fail,
                          cal_frac=args.cal_frac, mask=mask, risk_target="step")
    risk_hz = fit_risk_model(models, u_val, args.benchmark, p_val, dt, tau_fail=args.tau_fail,
                             cal_frac=args.cal_frac, mask=mask, risk_target="horizon",
                             gamma=args.gamma)
    risk_eval = evaluate_risk_on_split(models, u_test, args.benchmark, p_test, dt,
                                       risk["risk_model"], risk["conformal"],
                                       tau_fail=args.tau_fail, risk_target="step",
                                       risk_hz=risk_hz["risk_model"], conf_hz=risk_hz["conformal"])
    base = {
        "benchmark": args.benchmark, "seed": args.seed, "split": args.split,
        "n_test": n, "dt_save": dt, "model": args.model, "members": len(models),
        "risk_report": risk["report"], "risk_report_horizon": risk_hz["report"],
        "risk_eval": risk_eval,
        "prep_seconds": time.time() - t0,
    }
    with open(f"{out_dir}/{args.benchmark}_{args.split}_s{args.seed}_risk.json", "w") as f:
        json.dump(base, f, indent=2)
    print(f"[risk] AUROC cal={risk_eval['auroc_calibrated']:.3f} "
          f"res={risk_eval['auroc_residual']:.3f} ens={risk_eval['auroc_ensemble']:.3f} "
          f"fail={risk_eval['failure_rate']:.3f} ECE={risk_eval['ece']:.4f}", flush=True)

    # Gain-aware planning is only computed when requested: it needs a gain model
    # fitted on validation and, for the oracle, one counterfactual pass on test.
    gain_cache = {}
    if any(p in ("gain_ranked", "oracle_gain") for p in args.policies):
        gain = fit_gain_model(models, u_val, args.benchmark, p_val, dt, mask=mask)
        print(f"[gain] mean one-step gain={gain['mean_gain']:.4f} "
              f"relative={gain['mean_relative_gain']:.3f}", flush=True)
        t_gain = time.time()
        feats_g, errs_g, states_g = neural_pass(models, u_test, solver, p_test, dt,
                                                store_states=True)
        Xg = feats_g.reshape(-1, feats_g.shape[-1])
        if gain["mask"] is not None:
            Xg = Xg[:, np.asarray(gain["mask"], dtype=bool)]
        rel_gain = gain["gain_model"].predict_relative(Xg).reshape(feats_g.shape[:2])
        err_pred = risk["risk_model"].predict_error(
            feats_g.reshape(-1, feats_g.shape[-1])).reshape(feats_g.shape[:2])
        gain_cache["learned"] = rel_gain * err_pred
        gain_cache["plan_seconds"] = time.time() - t_gain
        if "oracle_gain" in args.policies:
            gain_cache["oracle"] = counterfactual_gains(states_g, u_test, args.benchmark,
                                                        p_test, dt)

    records = []
    for policy in args.policies:
        if policy in ("neural", "numerical"):
            actions = np.full((n, T), policy == "numerical", dtype=bool)
            res = execute_actions(models, u_test, actions, args.benchmark, p_test, dt, mode="step")
            rec = _record(policy, None, res)
            rec["variant"] = args.variant
            records.append(rec)
            print(f"{policy:20s} relL2={rec['rel_l2_mean']:.4f} cost={rec['cost_fraction']:.3f}", flush=True)
            continue

        if policy == "adaptive":
            for B in args.budgets:
                res = adaptive_execute(models, u_test, args.benchmark, p_test, dt, B,
                                       "residual", mode="step")
                rec = _record(policy, B, res)
                rec["variant"] = args.variant
                records.append(rec)
                print(f"{policy:20s} B={B:.2f} relL2={rec['rel_l2_mean']:.4f} "
                      f"cost={rec['cost_fraction']:.3f}", flush=True)
            continue

        if policy in ("uniform", "early"):
            for B in args.budgets:
                k = int(np.floor(B * T))
                actions = np.zeros((n, T), dtype=bool)
                if k > 0:
                    if policy == "early":
                        idx = np.arange(k)
                    else:
                        idx = np.unique(np.round(np.linspace(0, T - 1, k)).astype(int))
                    actions[:, idx] = True
                res = execute_actions(models, u_test, actions, args.benchmark, p_test, dt,
                                      mode="step")
                rec = _record(policy, B, res, {"schedule": policy, "n_corrected": int(k)})
                rec["variant"] = args.variant
                records.append(rec)
                print(f"{policy:20s} B={B:.2f} relL2={rec['rel_l2_mean']:.4f} "
                      f"cost={rec['cost_fraction']:.3f}", flush=True)
            continue

        if policy == "reanchor":
            for B in args.budgets:
                t_star = max(0, int(np.floor(B * T)) - 1)
                actions = np.zeros((n, T), dtype=bool)
                actions[:, t_star] = True
                res = execute_actions(models, u_test, actions, args.benchmark, p_test, dt,
                                      mode="reanchor")
                rec = _record(policy, B, res, {"reanchor_step": t_star, "mode": "reanchor"})
                rec["variant"] = args.variant
                records.append(rec)
                print(f"{policy:20s} B={B:.2f} relL2={rec['rel_l2_mean']:.4f} "
                      f"cost={rec['cost_fraction']:.3f} t*={t_star}", flush=True)
            continue

        # Static threshold on the raw residual feature, tuned on validation.
        if policy == "fixed":
            scores_val = risk["scores"]["residual"]
            scores_test = plan_scores(models, u_test, solver, p_test, dt, "residual")
            for B in args.budgets:
                tau = quantile_threshold(scores_val, B)
                actions = threshold_actions(scores_test, tau)
                res = execute_actions(models, u_test, actions, args.benchmark, p_test, dt)
                rec = _record(policy, B, res, {"tau": tau})
                rec["variant"] = args.variant
                records.append(rec)
                print(f"{policy:20s} B={B:.2f} relL2={rec['rel_l2_mean']:.4f} "
                      f"cost={rec['cost_fraction']:.3f}", flush=True)
            continue

        kind = SCORE_KIND[policy]
        if kind in ("gain", "oracle_gain"):
            scores = gain_cache["learned" if kind == "gain" else "oracle"]
            t_plan = time.time() - gain_cache.get("plan_seconds", 0.0)
        elif kind in ("horizon", "combo"):
            t_plan = time.time()
            feats, _ = neural_pass(models, u_test, solver, p_test, dt)
            hz = risk_hz["risk_model"].risk_score(
                feats.reshape(-1, feats.shape[-1]), mode=args.brace_mode).reshape(feats.shape[:2])
            if kind == "combo":
                # Fixed, a-priori equal-weight fusion of calibrated horizon risk
                # and ensemble disagreement, standardised on validation scores.
                hz_key = "brace" if args.brace_mode == "hybrid" else "calibrated"
                hz_val = np.asarray(risk_hz["scores"][hz_key])
                ens_val = np.asarray(risk["scores"]["ensemble"])
                z_hz = (hz - hz_val.mean()) / (hz_val.std() + 1e-8)
                z_ens = (feats[:, :, 0] - ens_val.mean()) / (ens_val.std() + 1e-8)
                scores = 0.5 * z_hz + 0.5 * z_ens
            else:
                scores = hz
        else:
            scores = plan_scores(models, u_test, solver, p_test, dt, kind, risk["risk_model"])
        if args.time_weight != "none":
            w = np.linspace(1.0, 0.0, T) if args.time_weight == "early" else np.linspace(0.0, 1.0, T)
            scores = scores * w[None, :]
        t_plan_end = time.time()
        for B in args.budgets:
            spacing = args.window if args.correction_mode == "window" else 1
            min_start = max(0, spacing - 1) if args.correction_mode == "window" else 0
            actions = topk_actions(scores, B, cost_per_action=spacing,
                                   min_spacing=spacing, min_start=min_start)
            res = execute_actions(models, u_test, actions, args.benchmark, p_test, dt,
                                  mode=args.correction_mode, window=args.window)
            rec = _record(policy, B, res, {"plan_seconds": t_plan_end - t_plan,
                                           "total_seconds": (t_plan_end - t_plan) + res["wall_seconds"],
                                           "mode": args.correction_mode})
            rec["variant"] = args.variant
            records.append(rec)
            print(f"{policy:20s} B={B:.2f} relL2={rec['rel_l2_mean']:.4f} "
                  f"cost={rec['cost_fraction']:.3f} fail={rec['step_failure_rate']:.3f}",
                  flush=True)

    suffix = args.correction_mode + (f"_w{args.window}" if args.correction_mode == "window" else "")
    if args.variant:
        suffix += f"_{args.variant}"
    out_path = f"{out_dir}/{args.benchmark}_{args.split}_s{args.seed}_{suffix}.json"
    if args.merge_records and os.path.exists(out_path):
        with open(out_path) as f:
            prev = json.load(f)["records"]
        key = lambda r: (r["policy"], r["budget"], r.get("variant", ""))
        keep = [r for r in prev if key(r) not in {key(r) for r in records}]
        records = keep + records
    with open(out_path, "w") as f:
        json.dump({"config": vars(args), "records": records}, f, indent=2)
    print(f"wrote {len(records)} records in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
