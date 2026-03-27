#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm

from utils.load_config import load_config
from utils.load_dataset import HitchDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Plot train GT angle distribution (orig vs aug).")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--bins", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def theta_deg_from_gt(gt):
    if isinstance(gt, torch.Tensor):
        gt = gt.detach().cpu().numpy()
    cos_g, sin_g = float(gt[0]), float(gt[1])
    return np.degrees(np.arctan2(sin_g, cos_g))


def build_dataset(cfg, split, aug_enabled, seed):
    dset_cfg = cfg["dataset"]
    train_cfg = cfg.get("train", {})

    if aug_enabled:
        aug_rotate_deg = train_cfg.get("aug_rotate_deg", 0.0)
        aug_rotate_prob = train_cfg.get("aug_rotate_prob", 0.0)
        aug_rotate_min_deg = train_cfg.get("aug_rotate_min_deg", None)
        aug_rotate_max_deg = train_cfg.get("aug_rotate_max_deg", None)
        aug_rotate_target_min_deg = train_cfg.get("aug_rotate_target_min_deg", None)
        aug_rotate_target_max_deg = train_cfg.get("aug_rotate_target_max_deg", None)
    else:
        aug_rotate_deg = 0.0
        aug_rotate_prob = 0.0
        aug_rotate_min_deg = None
        aug_rotate_max_deg = None
        aug_rotate_target_min_deg = None
        aug_rotate_target_max_deg = None

    np.random.seed(seed)
    return HitchDataset(
        root=dset_cfg["root"],
        split_json=dset_cfg["split"],
        split=split,
        temporal_window=dset_cfg.get("temporal_window", 20),
        micro_seq_length=dset_cfg.get("micro_seq_length", 10),
        trailer_type=dset_cfg.get("name", "charger"),
        normalize_xy=train_cfg.get("normalize_xy", False),
        bev_add_xy=train_cfg.get("bev_add_xy", False),
        bev_add_orient=train_cfg.get("bev_add_orient", False),
        aug_rotate_deg=aug_rotate_deg,
        aug_rotate_prob=aug_rotate_prob,
        aug_rotate_min_deg=aug_rotate_min_deg,
        aug_rotate_max_deg=aug_rotate_max_deg,
        aug_rotate_target_min_deg=aug_rotate_target_min_deg,
        aug_rotate_target_max_deg=aug_rotate_target_max_deg,
        bev_use_hmax=train_cfg.get("bev_use_hmax", True),
        bev_use_dlog=train_cfg.get("bev_use_dlog", True),
    )


def main():
    args = parse_args()
    cfg = load_config(args.config)

    orig_ds = build_dataset(cfg, "train", aug_enabled=False, seed=args.seed)
    aug_ds = build_dataset(cfg, "train", aug_enabled=True, seed=args.seed)

    max_samples = args.max_samples or len(orig_ds)
    stride = max(1, args.stride)

    orig_deg = []
    aug_deg = []

    for idx in tqdm(range(0, len(orig_ds), stride), desc="[Collect]"):
        if len(orig_deg) >= max_samples:
            break
        orig_sample = orig_ds[idx]
        aug_sample = aug_ds[idx]
        orig_deg.append(theta_deg_from_gt(orig_sample["gt"]))
        aug_deg.append(theta_deg_from_gt(aug_sample["gt"]))

    orig_deg = np.array(orig_deg, dtype=np.float32)
    aug_deg = np.array(aug_deg, dtype=np.float32)

    exp_cfg = cfg.get("experiment", {})
    exp_name = exp_cfg.get("name", os.path.splitext(os.path.basename(args.config))[0])
    out_dir = exp_cfg.get("output_dir", os.path.join("ckpts", exp_name))
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Fixed binning: [-60, 60] with 240 bins
    bins = np.linspace(-60.0, 60.0, num=241)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 4))
    plt.hist(orig_deg, bins=bins, alpha=0.6, label="orig")
    plt.hist(aug_deg, bins=bins, alpha=0.6, label="aug")
    plt.xlabel("Hitch Angle (deg)")
    plt.ylabel("Count")
    plt.title("Train GT Angle Distribution (orig vs aug)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"train_angle_hist_{timestamp}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()

    stats = {
        "bins_deg": bins.tolist(),
        "orig_count": np.histogram(orig_deg, bins=bins)[0].tolist(),
        "aug_count": np.histogram(aug_deg, bins=bins)[0].tolist(),
        "orig_samples": int(orig_deg.size),
        "aug_samples": int(aug_deg.size),
        "orig_mean": float(np.mean(orig_deg)) if orig_deg.size else 0.0,
        "aug_mean": float(np.mean(aug_deg)) if aug_deg.size else 0.0,
        "orig_std": float(np.std(orig_deg)) if orig_deg.size else 0.0,
        "aug_std": float(np.std(aug_deg)) if aug_deg.size else 0.0,
    }
    stats_path = os.path.join(out_dir, f"train_angle_hist_{timestamp}.json")
    with open(stats_path, "w") as f:
        import json
        json.dump(stats, f, indent=2)

    print(f"[INFO] Saved plot -> {out_path}")
    print(f"[INFO] Saved stats -> {stats_path}")
    print(f"[INFO] orig samples: {orig_deg.size} | aug samples: {aug_deg.size}")


if __name__ == "__main__":
    main()
