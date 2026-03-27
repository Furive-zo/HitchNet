#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
from pathlib import Path

import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.load_config import load_config
from utils.load_dataset import HitchDataset
from utils.collate import collate_fn
from utils.angle import wrap_rad_torch
from models import build_model


SCENARIO_GROUPS = {
    "low curvature": {"highway_5", "urban_5"},
    "high curvature": {"reverse_4", "scurve_5"},
    "elevation": {"updown_9"},
}

DISPLAY_LABELS = {
    "low curvature": "low curvature",
    "high curvature": "high curvature",
    "elevation": "elevated segment (inclined/banked)",
}

DEFAULT_MODELS = [
    ("Naive BEV", "configs/experiments/dummy_bev_resnet_regression.yaml", None, "#1F4E79"),
    ("CORAL-DG", "configs/experiments/dummy_temporary_bev_resnet_regression_coral_dg.yaml", None, "#7A5EA8"),
    ("CORAL-UDA", "configs/experiments/dummy_bev_resnet_regression_coral_charger.yaml", None, "#2E8B57"),
    ("Ours(Alignment only)", "configs/experiments/dummy_bev_resnet_regression_norm.yaml", None, "#F4A261"),
    ("Ours(Full)", "configs/experiments/dummy_bev_resnet_regression_norm_aug.yaml", None, "#D62828"),
]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"],
    "font.size": 12,
})


def parse_args():
    parser = argparse.ArgumentParser(description="Scenario-wise RMSE comparison plotter")
    parser.add_argument(
        "--models",
        type=str,
        nargs="*",
        default=None,
        help=(
            "Model specs. Format: name|config|ckpt|color. "
            "ckpt can be '-' to use default best.pth. If omitted, uses default model set."
        ),
    )
    parser.add_argument(
        "--eval_trailer_type",
        type=str,
        choices=["charger", "dummy", "temporary"],
        default="charger",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--out_png", type=str, default="results/scenario_rmse_compare.png")
    parser.add_argument("--out_json", type=str, default="results/scenario_rmse_compare.json")
    return parser.parse_args()


def move_batch_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
    return out


def get_group_name(seq_name):
    for group, seqs in SCENARIO_GROUPS.items():
        if seq_name in seqs:
            return group
    return None


def parse_models(model_specs):
    if not model_specs:
        return DEFAULT_MODELS

    models = []
    for spec in model_specs:
        parts = spec.split("|")
        if len(parts) != 4:
            raise ValueError(f"Invalid model spec: {spec}")
        name, config, ckpt, color = parts
        ckpt = None if ckpt.strip() == "-" else ckpt.strip()
        models.append((name.strip(), config.strip(), ckpt, color.strip()))
    return models


def resolve_eval_dataset(cfg_path, eval_trailer_type):
    cfg_dir = os.path.dirname(cfg_path)
    ds_rel = f"datasets/{eval_trailer_type}.yaml"
    ds_path = os.path.normpath(os.path.join(cfg_dir, "..", ds_rel))
    with open(ds_path, "r") as f:
        ds_cfg = yaml.safe_load(f)
    return ds_cfg.get("dataset", {})


def build_dataset_loader(cfg, cfg_path, eval_trailer_type, batch_size_override, num_workers_override):
    dset_cfg = resolve_eval_dataset(cfg_path, eval_trailer_type)
    train_cfg = cfg.get("train", {})

    dataset = HitchDataset(
        root=dset_cfg["root"],
        split_json=dset_cfg["split"],
        split="test",
        temporal_window=dset_cfg.get("temporal_window", 20),
        micro_seq_length=dset_cfg.get("micro_seq_length", 10),
        trailer_type=dset_cfg.get("name", eval_trailer_type),
        normalize_xy=train_cfg.get("normalize_xy", False),
        bev_add_xy=train_cfg.get("bev_add_xy", False),
        bev_add_orient=train_cfg.get("bev_add_orient", False),
        bev_use_hmax=train_cfg.get("bev_use_hmax", True),
        bev_use_dlog=train_cfg.get("bev_use_dlog", True),
        occ_binary=train_cfg.get("occ_binary", False),
        add_observed_mask=train_cfg.get("add_observed_mask", False),
        observed_bins=int(train_cfg.get("observed_bins", 360)),
        observed_margin=float(train_cfg.get("observed_margin", 0.0)),
        centroid_mode=train_cfg.get("centroid_mode", "minmax"),
        dlog_range_norm=train_cfg.get("dlog_range_norm", False),
        dlog_range_norm_mode=train_cfg.get("dlog_range_norm_mode", "center"),
        joint_shift_x=float(train_cfg.get("joint_shift_x", 0.8)),
    )

    exp_name = cfg.get("experiment", {}).get("name", Path(cfg_path).stem)
    if bool(train_cfg.get("dlog_range_norm", False)):
        bin_size = float(train_cfg.get("dlog_range_bin_size", 0.1))
        stats_path = train_cfg.get(
            "dlog_stats_path",
            os.path.join("ckpts", exp_name, f"dlog_range_stats_b{bin_size}.pt"),
        )
        if os.path.exists(stats_path):
            stats = torch.load(stats_path, map_location="cpu", weights_only=False)
            dataset.dlog_range_stats = stats
            dataset._build_range_bin_idx()

    num_workers = num_workers_override if num_workers_override is not None else dset_cfg.get("num_workers", 4)
    batch_size = batch_size_override if batch_size_override is not None else train_cfg.get("batch_size", 8)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    return dataset, loader


def evaluate_model(model_name, cfg_path, ckpt_override, device, eval_trailer_type, batch_size, num_workers):
    cfg = load_config(cfg_path)
    model_cfg = cfg["model"]
    exp_name = cfg.get("experiment", {}).get("name", Path(cfg_path).stem)
    ckpt_path = ckpt_override or os.path.join("ckpts", exp_name, "best.pth")

    dataset, loader = build_dataset_loader(cfg, cfg_path, eval_trailer_type, batch_size, num_workers)

    model = build_model(model_cfg).to(device)
    if model_cfg.get("name") not in ("rule_based", "rule_based_pca", "rule_based_ols", "rule_based_mle"):
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"[{model_name}] checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt, strict=True)
    model.eval()

    sq_err_by_group = {k: [] for k in SCENARIO_GROUPS.keys()}
    idx_global = 0
    with torch.no_grad():
        for batch in loader:
            bsz = batch["gt"].shape[0]
            batch = move_batch_to_device(batch, device)
            pred = model(batch)
            gt = batch["gt"]

            theta_p = torch.atan2(pred[:, 1], pred[:, 0])
            theta_g = torch.atan2(gt[:, 1], gt[:, 0])
            err_deg = wrap_rad_torch(theta_p - theta_g) * 180.0 / np.pi
            err_np = err_deg.detach().cpu().numpy()

            for i in range(bsz):
                fr = dataset.frame_dirs[idx_global + i]
                seq_name = Path(fr).parent.name
                group = get_group_name(seq_name)
                if group is not None:
                    sq_err_by_group[group].append(float(err_np[i] ** 2))
            idx_global += bsz

    order = ["low curvature", "high curvature", "elevation"]
    rmse_vals = []
    counts = []
    for g in order:
        vals = sq_err_by_group[g]
        counts.append(len(vals))
        rmse_vals.append(float(np.sqrt(np.mean(vals))) if len(vals) > 0 else float("nan"))
    return ckpt_path, order, rmse_vals, counts


def main():
    args = parse_args()
    model_specs = parse_models(args.models)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    results = []
    for model_name, cfg_path, ckpt_override, color in model_specs:
        ckpt_path, order, rmse_vals, counts = evaluate_model(
            model_name=model_name,
            cfg_path=cfg_path,
            ckpt_override=ckpt_override,
            device=device,
            eval_trailer_type=args.eval_trailer_type,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        results.append({
            "name": model_name,
            "config": cfg_path,
            "checkpoint": ckpt_path,
            "color": color,
            "order": order,
            "rmse_vals": rmse_vals,
            "counts": counts,
        })

    order = ["low curvature", "high curvature", "elevation"]
    xlabels = [DISPLAY_LABELS[k] for k in order]
    x = np.arange(len(order))
    n_models = len(results)
    total_width = 0.78
    bar_w = total_width / max(n_models, 1)

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for i, r in enumerate(results):
        offset = -total_width / 2 + (i + 0.5) * bar_w
        xs = x + offset
        bars = ax.bar(
            xs,
            r["rmse_vals"],
            width=bar_w * 0.95,
            color=r["color"],
            edgecolor="#1f1f1f",
            linewidth=0.7,
            label=r["name"],
        )
        for j, b in enumerate(bars):
            y = r["rmse_vals"][j]
            if np.isfinite(y):
                ax.text(b.get_x() + b.get_width() / 2.0, y, f"{y:.2f}",
                        ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.set_ylabel("RMSE (°)")
    ax.set_xlabel("")
    ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Put legend above axes to avoid covering bars.
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=max(len(results), 1),
        frameon=False,
        handlelength=1.4,
        columnspacing=1.0,
    )

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.93])
    out_dir = os.path.dirname(args.out_png)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(args.out_png, dpi=220)
    plt.close(fig)

    payload = {
        "eval_trailer_type": args.eval_trailer_type,
        "scenario_order": order,
        "scenario_labels": {k: DISPLAY_LABELS[k] for k in order},
        "scenario_groups": {k: sorted(list(v)) for k, v in SCENARIO_GROUPS.items()},
        "models": [],
    }
    for r in results:
        payload["models"].append({
            "name": r["name"],
            "config": r["config"],
            "checkpoint": r["checkpoint"],
            "color": r["color"],
            "rmse_deg": {k: v for k, v in zip(order, r["rmse_vals"])},
            "counts": {k: v for k, v in zip(order, r["counts"])},
        })

    out_json_dir = os.path.dirname(args.out_json)
    if out_json_dir:
        os.makedirs(out_json_dir, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"[INFO] Saved plot: {args.out_png}")
    print(f"[INFO] Saved metrics: {args.out_json}")
    for r in results:
        print(f"[INFO] Model: {r['name']}")
        for g, rmse, n in zip(order, r["rmse_vals"], r["counts"]):
            print(f"       {g:14s} RMSE={rmse:.3f} deg (n={n})")


if __name__ == "__main__":
    main()
