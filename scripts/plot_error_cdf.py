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
    parser = argparse.ArgumentParser(description="Error distribution CDF plotter")
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
        help="Evaluation dataset trailer type (default: charger).",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--x_max", type=float, default=15.0, help="Main CDF x-axis max (deg).")
    parser.add_argument("--zoom_max", type=float, default=3.0, help="Zoomed inset x-axis max (deg).")
    parser.add_argument("--out_png", type=str, default="results/error_cdf_compare.png")
    parser.add_argument("--out_json", type=str, default="results/error_cdf_compare.json")
    return parser.parse_args()


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


def move_batch_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
    return out


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
    return loader


def evaluate_abs_errors(model_name, cfg_path, ckpt_override, device, eval_trailer_type, batch_size, num_workers):
    cfg = load_config(cfg_path)
    model_cfg = cfg["model"]
    exp_name = cfg.get("experiment", {}).get("name", Path(cfg_path).stem)
    ckpt_path = ckpt_override or os.path.join("ckpts", exp_name, "best.pth")

    loader = build_dataset_loader(cfg, cfg_path, eval_trailer_type, batch_size, num_workers)
    model = build_model(model_cfg).to(device)
    if model_cfg.get("name") not in ("rule_based", "rule_based_pca", "rule_based_ols", "rule_based_mle"):
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"[{model_name}] checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt, strict=True)
    model.eval()

    abs_errs = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            pred = model(batch)
            gt = batch["gt"]
            theta_p = torch.atan2(pred[:, 1], pred[:, 0])
            theta_g = torch.atan2(gt[:, 1], gt[:, 0])
            err_deg = wrap_rad_torch(theta_p - theta_g) * 180.0 / np.pi
            abs_errs.append(torch.abs(err_deg).detach().cpu().numpy())

    if len(abs_errs) == 0:
        return ckpt_path, np.array([], dtype=np.float32)
    return ckpt_path, np.concatenate(abs_errs, axis=0).astype(np.float32)


def ecdf(values):
    if values.size == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
    x = np.sort(values)
    y = np.arange(1, x.size + 1, dtype=np.float32) / float(x.size)
    return x, y


def main():
    args = parse_args()
    models = parse_models(args.models)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    results = []
    for model_name, cfg_path, ckpt_override, color in models:
        ckpt_path, abs_err = evaluate_abs_errors(
            model_name=model_name,
            cfg_path=cfg_path,
            ckpt_override=ckpt_override,
            device=device,
            eval_trailer_type=args.eval_trailer_type,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        abs_err_clip = np.clip(abs_err, 0.0, float(args.x_max))
        x, y = ecdf(abs_err_clip)
        results.append({
            "name": model_name,
            "config": cfg_path,
            "checkpoint": ckpt_path,
            "color": color,
            "abs_err": abs_err,
            "x": x,
            "y": y,
        })

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for r in results:
        ax.plot(r["x"], r["y"], color=r["color"], linewidth=2.0, label=r["name"])

    ax.set_xlim(0.0, float(args.x_max))
    ax.set_ylim(0.0, 1.01)
    ax.set_xlabel("Absolute Error (°)")
    ax.set_ylabel("Probability")
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(frameon=False, loc="lower right")

    fig.tight_layout()
    out_png_dir = os.path.dirname(args.out_png)
    if out_png_dir:
        os.makedirs(out_png_dir, exist_ok=True)
    fig.savefig(args.out_png, dpi=220)
    plt.close(fig)

    payload = {
        "eval_trailer_type": args.eval_trailer_type,
        "x_range_deg": [0.0, float(args.x_max)],
        "zoom_range_deg": [0.0, float(args.zoom_max)],
        "models": [],
    }
    for r in results:
        payload["models"].append({
            "name": r["name"],
            "config": r["config"],
            "checkpoint": r["checkpoint"],
            "color": r["color"],
            "count": int(r["abs_err"].size),
            "rmse_deg": float(np.sqrt(np.mean(r["abs_err"] ** 2))) if r["abs_err"].size > 0 else float("nan"),
            "mae_deg": float(np.mean(r["abs_err"])) if r["abs_err"].size > 0 else float("nan"),
            "p95_deg": float(np.percentile(r["abs_err"], 95)) if r["abs_err"].size > 0 else float("nan"),
        })

    out_json_dir = os.path.dirname(args.out_json)
    if out_json_dir:
        os.makedirs(out_json_dir, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"[INFO] Saved plot: {args.out_png}")
    print(f"[INFO] Saved metrics: {args.out_json}")


if __name__ == "__main__":
    main()
