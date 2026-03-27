#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from pathlib import Path
from datetime import datetime
import time
import json
import csv
import math
import random
import yaml
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler, ConcatDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

try:
    from torch.amp import autocast as _amp_autocast
    from torch.amp import GradScaler as _AmpGradScaler
    _USE_TORCH_AMP = True
except ImportError:
    from torch.cuda.amp import autocast as _amp_autocast
    from torch.cuda.amp import GradScaler as _AmpGradScaler
    _USE_TORCH_AMP = False

from utils.load_config import load_config
from utils.load_dataset import HitchDataset
from utils.collate import collate_fn
from utils.loss import hitch_loss
from utils.angle import wrap_rad_torch, wrap_deg_torch

from models import build_model


@contextmanager
def amp_autocast_cuda(enabled: bool, dtype=None):
    if _USE_TORCH_AMP:
        with _amp_autocast("cuda", enabled=enabled, dtype=dtype):
            yield
    else:
        kwargs = {"enabled": enabled}
        if dtype is not None:
            kwargs["dtype"] = dtype
        with _amp_autocast(**kwargs):
            yield


def build_grad_scaler(enabled: bool):
    if _USE_TORCH_AMP:
        return _AmpGradScaler("cuda", enabled=enabled)
    return _AmpGradScaler(enabled=enabled)


def parse_args():
    parser = argparse.ArgumentParser(description="HitchNet training script")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=None)
    return parser.parse_args()

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def move_batch_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
    return out


def _load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _resolve_cfg_path(exp_cfg_path, rel_or_abs):
    if os.path.isabs(rel_or_abs):
        return rel_or_abs
    base_dir = os.path.dirname(exp_cfg_path)
    return os.path.normpath(os.path.join(base_dir, "..", rel_or_abs))

def compute_or_load_angle_bins(train_dataset, out_dir, K=24):
    """
    train_dataset에서 gt(cos,sin)로 theta 분포를 세어 bin_counts 생성.
    처음 1회만 계산하고 out_dir에 캐시 저장.
    """
    cache_path = os.path.join(out_dir, f"angle_bins_K{K}.pt")
    if os.path.isfile(cache_path):
        obj = torch.load(cache_path, map_location="cpu")
        return obj["bins"], obj["bin_counts"]

    bins = torch.linspace(-math.pi/2, math.pi/2, steps=K + 1)  # (K+1,)
    bin_counts = torch.zeros(K, dtype=torch.long)

    for i in tqdm(range(len(train_dataset)), desc="[Init] building angle bin_counts"):
        sample = train_dataset[i]
        gt = sample["gt"]  # (2,) (cos,sin)
        # gt가 torch/np 둘 다 대응
        if isinstance(gt, torch.Tensor):
            cos_g = float(gt[0].item())
            sin_g = float(gt[1].item())
        else:
            cos_g = float(gt[0])
            sin_g = float(gt[1])

        theta = math.atan2(sin_g, cos_g)
        # bucketize는 torch 텐서 입력이 필요
        theta_t = torch.tensor(theta)
        bid = torch.bucketize(theta_t, bins, right=False).item() - 1
        bid = max(0, min(K - 1, bid))
        bin_counts[bid] += 1

    torch.save({"bins": bins, "bin_counts": bin_counts}, cache_path)
    return bins, bin_counts


def compute_or_load_angle_weights(train_dataset, out_dir, K=36, alpha=0.5, max_w=3.0):
    """
    Build per-sample weights for oversampling tails without down-weighting center bins.
    weights[i] >= 1.0 always, capped by max_w.
    """
    cache_path = os.path.join(out_dir, f"angle_weights_K{K}_a{alpha}_m{max_w}.pt")
    if os.path.exists(cache_path):
        obj = torch.load(cache_path)
        return obj["bins"], obj["bin_counts"], obj["weights"]

    bins, bin_counts = compute_or_load_angle_bins(train_dataset, out_dir, K=K)
    bin_counts = bin_counts.float()
    max_count = float(bin_counts.max().item()) if bin_counts.numel() else 1.0

    weights = torch.zeros(len(train_dataset), dtype=torch.float32)
    for i in tqdm(range(len(train_dataset)), desc="[Init] building angle weights"):
        sample = train_dataset[i]
        gt = sample["gt"]
        theta = torch.atan2(gt[1], gt[0])
        bid = torch.bucketize(theta, bins, right=False).item() - 1
        if bid < 0 or bid >= len(bin_counts):
            w = 1.0
        else:
            cnt = float(bin_counts[bid].item())
            if cnt <= 0:
                w = max_w
            else:
                ratio = max_count / cnt
                w = max(1.0, ratio ** alpha)
                w = min(w, max_w)
        weights[i] = w

    torch.save({"bins": bins, "bin_counts": bin_counts, "weights": weights}, cache_path)
    return bins, bin_counts, weights


def compute_or_load_dlog_range_stats(train_dataset, out_dir, bin_size=0.1, max_samples=None, stride=1):
    cache_path = os.path.join(out_dir, f"dlog_range_stats_b{bin_size}.pt")
    if os.path.exists(cache_path):
        return torch.load(cache_path, map_location="cpu", weights_only=False)

    x_min, x_max = train_dataset.bev_x_range
    y_min, y_max = train_dataset.bev_y_range
    res = train_dataset.bev_res
    H = int(np.ceil((x_max - x_min) / res))
    W = int(np.ceil((y_max - y_min) / res))
    x_centers = x_min + (np.arange(H) + 0.5) * res
    y_centers = y_min + (np.arange(W) + 0.5) * res
    x_grid, y_grid = np.meshgrid(x_centers, y_centers, indexing="ij")
    r = np.sqrt(x_grid ** 2 + y_grid ** 2)
    nbins = int(np.ceil(r.max() / bin_size)) + 1
    bin_idx = np.floor(r / bin_size).astype(np.int64)
    bin_idx = np.clip(bin_idx, 0, nbins - 1)

    sum_bins = np.zeros(nbins, dtype=np.float64)
    sumsq_bins = np.zeros(nbins, dtype=np.float64)
    count_bins = np.zeros(nbins, dtype=np.int64)

    indices = list(range(0, len(train_dataset), max(stride, 1)))
    if max_samples is not None:
        indices = indices[:max_samples]

    for i in tqdm(indices, desc="[Init] building dlog range stats"):
        item = train_dataset[i]
        bev = item["bev"].numpy()
        dlog_idx = 1 + (1 if train_dataset.bev_use_hmax else 0)
        dlog = bev[dlog_idx]
        for b in range(nbins):
            m = bin_idx == b
            vals = dlog[m]
            sum_bins[b] += vals.sum()
            sumsq_bins[b] += (vals ** 2).sum()
            count_bins[b] += vals.size

    mean = sum_bins / np.maximum(count_bins, 1)
    var = sumsq_bins / np.maximum(count_bins, 1) - mean ** 2
    std = np.sqrt(np.maximum(var, 1e-9))

    stats = {
        "bin_size": float(bin_size),
        "nbins": int(nbins),
        "mu": mean.astype(np.float32),
        "sigma": std.astype(np.float32),
        "count": count_bins.astype(np.int64),
    }
    torch.save(stats, cache_path)
    return stats


def save_aug_angle_hist(cfg, out_dir, seed=42):
    train_cfg = cfg.get("train", {})
    if not bool(train_cfg.get("save_aug_hist", True)):
        return
    dset_cfg = cfg["dataset"]

    aug_rotate_deg = float(train_cfg.get("aug_rotate_deg", 0.0))
    aug_rotate_prob = float(train_cfg.get("aug_rotate_prob", 0.0))
    aug_rotate_min_deg = train_cfg.get("aug_rotate_min_deg", None)
    aug_rotate_max_deg = train_cfg.get("aug_rotate_max_deg", None)
    aug_rotate_target_min_deg = train_cfg.get("aug_rotate_target_min_deg", None)
    aug_rotate_target_max_deg = train_cfg.get("aug_rotate_target_max_deg", None)

    if (
        aug_rotate_prob <= 0.0
        and aug_rotate_deg == 0.0
        and aug_rotate_min_deg is None
        and aug_rotate_max_deg is None
        and aug_rotate_target_min_deg is None
        and aug_rotate_target_max_deg is None
    ):
        return

    occ_prob = train_cfg.get("occ_prob", 0.0)
    occ_x_thresh = train_cfg.get("occ_x_thresh", -0.3)
    occ_box_x = train_cfg.get("occ_box_x", 0.0)
    occ_box_y = train_cfg.get("occ_box_y", 0.0)
    trans_prob = train_cfg.get("trans_prob", 0.0)
    trans_max_m = train_cfg.get("trans_max_m", 0.0)
    trans_dir = train_cfg.get("trans_dir", "both")

    def build_dataset(aug_enabled: bool):
        if aug_enabled:
            a_deg = aug_rotate_deg
            a_prob = aug_rotate_prob
            a_min = aug_rotate_min_deg
            a_max = aug_rotate_max_deg
            a_tmin = aug_rotate_target_min_deg
            a_tmax = aug_rotate_target_max_deg
        else:
            a_deg = 0.0
            a_prob = 0.0
            a_min = None
            a_max = None
            a_tmin = None
            a_tmax = None

        np.random.seed(seed)
        return HitchDataset(
            root=dset_cfg["root"],
            split_json=dset_cfg["split"],
            split="train",
            temporal_window=dset_cfg.get("temporal_window", 20),
            micro_seq_length=dset_cfg.get("micro_seq_length", 10),
            trailer_type=dset_cfg.get("name", "charger"),
            normalize_xy=train_cfg.get("normalize_xy", False),
            bev_add_xy=train_cfg.get("bev_add_xy", False),
            bev_add_orient=train_cfg.get("bev_add_orient", False),
            aug_rotate_deg=a_deg,
            aug_rotate_prob=a_prob,
            aug_rotate_min_deg=a_min,
            aug_rotate_max_deg=a_max,
            aug_rotate_target_min_deg=a_tmin,
            aug_rotate_target_max_deg=a_tmax,
            occ_prob=occ_prob if aug_enabled else 0.0,
            occ_x_thresh=occ_x_thresh,
            occ_box_x=occ_box_x,
            occ_box_y=occ_box_y,
            trans_prob=trans_prob if aug_enabled else 0.0,
            trans_max_m=trans_max_m,
            trans_dir=trans_dir,
        )

    def theta_deg_from_gt(gt):
        if isinstance(gt, torch.Tensor):
            gt = gt.detach().cpu().numpy()
        return float(np.degrees(np.arctan2(gt[1], gt[0])))

    orig_ds = build_dataset(aug_enabled=False)
    aug_ds = build_dataset(aug_enabled=True)

    max_samples = train_cfg.get("aug_hist_max_samples", None)
    stride = int(train_cfg.get("aug_hist_stride", 1))
    stride = max(1, stride)
    if max_samples is None:
        max_samples = len(orig_ds)
    max_samples = int(max_samples)

    orig_deg = []
    aug_deg = []
    for idx in tqdm(range(0, len(orig_ds), stride), desc="[Init] building aug angle hist"):
        if len(orig_deg) >= max_samples:
            break
        orig_deg.append(theta_deg_from_gt(orig_ds[idx]["gt"]))
        aug_deg.append(theta_deg_from_gt(aug_ds[idx]["gt"]))

    orig_deg = np.array(orig_deg, dtype=np.float32)
    aug_deg = np.array(aug_deg, dtype=np.float32)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
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
        json.dump(stats, f, indent=2)

    print(f"[INFO] Saved aug hist -> {out_path}")
    print(f"[INFO] Saved aug stats -> {stats_path}")


def load_bev_pretrained(model, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    if hasattr(model, "bev_enc"):
        if any(k.startswith("bev_enc.") for k in state):
            sub = {k[len("bev_enc."):]: v for k, v in state.items() if k.startswith("bev_enc.")}
        elif any(k.startswith("encoder.") for k in state):
            sub = {k[len("encoder."):]: v for k, v in state.items() if k.startswith("encoder.")}
        else:
            sub = {k: v for k, v in state.items() if k.startswith(("stem.", "layer", "pool", "proj"))}
        missing, unexpected = model.bev_enc.load_state_dict(sub, strict=False)
        return missing, unexpected

    if hasattr(model, "encoder"):
        sub = {k[len("encoder."):]: v for k, v in state.items() if k.startswith("encoder.")}
        missing, unexpected = model.encoder.load_state_dict(sub, strict=False)
        return missing, unexpected

    raise ValueError("Model has no BEV encoder to load pretrained weights.")

def main():
    args = parse_args()

    # Workaround for environments with mismatched cuDNN train/infer shared libs.
    # Enable with: HITCHNET_DISABLE_CUDNN=1
    if os.environ.get("HITCHNET_DISABLE_CUDNN", "0") == "1":
        torch.backends.cudnn.enabled = False
        print("[INFO] cuDNN disabled by HITCHNET_DISABLE_CUDNN=1")

    # ============================
    # 1) Config
    # ============================
    cfg = load_config(args.config)
    raw_exp = _load_yaml(args.config)
    exp_cfg = cfg.get("experiment", {})
    dset_cfg = cfg["dataset"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]

    exp_name = exp_cfg.get("name", Path(args.config).stem)
    out_dir = exp_cfg.get("output_dir", os.path.join("ckpts", exp_name))
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    seed = int(exp_cfg.get("seed", 42))
    set_global_seed(seed)
    print(f"[INFO] Seed set to: {seed}")

    # ============================
    # 2) Dataset & Loader
    # ============================
    num_workers = args.num_workers or dset_cfg.get("num_workers", 4)
    batch_size = train_cfg.get("batch_size", 8)

    root = dset_cfg["root"]
    split_json = dset_cfg["split"]

    temporal_window = dset_cfg.get("temporal_window", 20)
    micro_seq_length = dset_cfg.get("micro_seq_length", 10)
    trailer_type = dset_cfg.get("name", "charger")
    normalize_xy = train_cfg.get("normalize_xy", False)
    bev_add_xy = train_cfg.get("bev_add_xy", False)
    bev_add_orient = train_cfg.get("bev_add_orient", False)
    bev_use_hmax = train_cfg.get("bev_use_hmax", True)
    bev_use_dlog = train_cfg.get("bev_use_dlog", True)
    occ_binary = train_cfg.get("occ_binary", False)
    add_observed_mask = train_cfg.get("add_observed_mask", False)
    observed_bins = int(train_cfg.get("observed_bins", 360))
    observed_margin = float(train_cfg.get("observed_margin", 0.0))
    dlog_range_norm = train_cfg.get("dlog_range_norm", False)
    dlog_range_norm_mode = train_cfg.get("dlog_range_norm_mode", "center")
    dlog_range_bin_size = float(train_cfg.get("dlog_range_bin_size", 0.1))
    dlog_range_max_samples = train_cfg.get("dlog_range_max_samples", None)
    dlog_range_stride = train_cfg.get("dlog_range_stride", 1)
    centroid_mode = train_cfg.get("centroid_mode", "minmax")
    aug_rotate_deg = train_cfg.get("aug_rotate_deg", 0.0)
    aug_rotate_prob = train_cfg.get("aug_rotate_prob", 0.0)
    aug_rotate_min_deg = train_cfg.get("aug_rotate_min_deg", None)
    aug_rotate_max_deg = train_cfg.get("aug_rotate_max_deg", None)
    aug_rotate_target_min_deg = train_cfg.get("aug_rotate_target_min_deg", None)
    aug_rotate_target_max_deg = train_cfg.get("aug_rotate_target_max_deg", None)
    occ_prob = train_cfg.get("occ_prob", 0.0)
    occ_x_thresh = train_cfg.get("occ_x_thresh", -0.3)
    occ_box_x = train_cfg.get("occ_box_x", 0.0)
    occ_box_y = train_cfg.get("occ_box_y", 0.0)
    trans_prob = train_cfg.get("trans_prob", 0.0)
    trans_max_m = train_cfg.get("trans_max_m", 0.0)
    trans_dir = train_cfg.get("trans_dir", "both")
    aug_vis = bool(train_cfg.get("aug_vis", False))
    aug_vis_max = int(train_cfg.get("aug_vis_max", 8))
    joint_shift_x = float(train_cfg.get("joint_shift_x", 0.8))
    source_dataset_cfgs = raw_exp.get("source_dataset_configs", None)
    if source_dataset_cfgs:
        resolved_sources = []
        for p in source_dataset_cfgs:
            ap = _resolve_cfg_path(args.config, p)
            resolved_sources.append(_load_yaml(ap)["dataset"])
    else:
        resolved_sources = [dset_cfg]

    def _build_split_dataset(ds_cfg, split: str, with_aug: bool):
        return HitchDataset(
            root=ds_cfg["root"],
            split_json=ds_cfg["split"],
            split=split,
            temporal_window=ds_cfg.get("temporal_window", temporal_window),
            micro_seq_length=ds_cfg.get("micro_seq_length", micro_seq_length),
            trailer_type=ds_cfg.get("name", trailer_type),
            normalize_xy=normalize_xy,
            bev_add_xy=bev_add_xy,
            bev_add_orient=bev_add_orient,
            bev_use_hmax=bev_use_hmax,
            bev_use_dlog=bev_use_dlog,
            occ_binary=occ_binary,
            add_observed_mask=add_observed_mask,
            observed_bins=observed_bins,
            observed_margin=observed_margin,
            centroid_mode=centroid_mode,
            aug_rotate_deg=aug_rotate_deg if with_aug else 0.0,
            aug_rotate_prob=aug_rotate_prob if with_aug else 0.0,
            aug_rotate_min_deg=aug_rotate_min_deg if with_aug else None,
            aug_rotate_max_deg=aug_rotate_max_deg if with_aug else None,
            aug_rotate_target_min_deg=aug_rotate_target_min_deg if with_aug else None,
            aug_rotate_target_max_deg=aug_rotate_target_max_deg if with_aug else None,
            occ_prob=occ_prob if with_aug else 0.0,
            occ_x_thresh=occ_x_thresh,
            occ_box_x=occ_box_x,
            occ_box_y=occ_box_y,
            trans_prob=trans_prob if with_aug else 0.0,
            trans_max_m=trans_max_m,
            trans_dir=trans_dir,
            joint_shift_x=joint_shift_x,
            aug_vis=aug_vis if with_aug else False,
            dlog_range_norm=dlog_range_norm,
            dlog_range_norm_mode=dlog_range_norm_mode,
        )

    train_sets = [_build_split_dataset(ds, "train", with_aug=True) for ds in resolved_sources]
    val_sets = [_build_split_dataset(ds, "val", with_aug=False) for ds in resolved_sources]
    base_train_dataset = train_sets[0]

    train_dataset = train_sets[0] if len(train_sets) == 1 else ConcatDataset(train_sets)
    val_dataset = val_sets[0] if len(val_sets) == 1 else ConcatDataset(val_sets)

    if len(train_sets) == 1:
        save_aug_angle_hist(cfg, out_dir, seed=seed)
    else:
        print(f"[INFO] Multi-source mode: {len(train_sets)} sources (aug histogram skipped).")

    if dlog_range_norm:
        if isinstance(train_dataset, ConcatDataset):
            raise ValueError("dlog_range_norm with source_dataset_configs is not supported in scripts/train.py yet.")
        stats = compute_or_load_dlog_range_stats(
            train_dataset,
            out_dir,
            bin_size=dlog_range_bin_size,
            max_samples=dlog_range_max_samples,
            stride=dlog_range_stride,
        )
        train_dataset.dlog_range_stats = stats
        train_dataset._build_range_bin_idx()
        val_dataset.dlog_range_stats = stats
        val_dataset._build_range_bin_idx()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        drop_last=True,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        drop_last=False,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed),
    )

    # ====== angle-bin reweight 준비 ======
    K = train_cfg.get("angle_bins_K", 36)
    bin_alpha = train_cfg.get("bin_alpha", 0.0)  # 0.0이면 OFF
    # ====== oversample (tails only) ======
    oversample_alpha = train_cfg.get("oversample_alpha", 0.0)
    oversample_max_w = train_cfg.get("oversample_max_w", 3.0)
    oversample_K = train_cfg.get("oversample_K", K)
    if oversample_alpha > 0.0:
        _, _, sample_weights = compute_or_load_angle_weights(
            train_dataset, out_dir, K=oversample_K, alpha=oversample_alpha, max_w=oversample_max_w
        )
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    else:
        sampler = None

    if bin_alpha > 0.0:
        bins, bin_counts = compute_or_load_angle_bins(train_dataset, out_dir, K=K)
        bins = bins.to(device)
        bin_counts = bin_counts.to(device)
    else:
        bins, bin_counts = None, None
    line_loss_w = train_cfg.get("line_loss_w", 0.0)
    bbox_loss_w = train_cfg.get("bbox_loss_w", 0.0)
    loss_alpha = train_cfg.get("loss_alpha", 1.0)
    loss_beta = train_cfg.get("loss_beta", 0.0)
    weight_factor = train_cfg.get("weight_factor", 3.0)
    huber_delta_deg = train_cfg.get("huber_delta_deg", 0.5)
    centroid_debug = bool(train_cfg.get("centroid_debug", False))
    centroid_debug_max = int(train_cfg.get("centroid_debug_max", 5))
    bbox_len = None
    bbox_wid = None
    if bbox_loss_w > 0.0:
        from utils.load_dataset import TRAILER_TYPES
        trailer_type = dset_cfg.get("name", "charger")
        bbox_len_scale = float(train_cfg.get("bbox_len_scale", 1.2))
        bbox_wid_scale = float(train_cfg.get("bbox_wid_scale", 1.0))
        if train_cfg.get("normalize_xy", False):
            bbox_len = 1.0 * bbox_len_scale
            bbox_wid = 1.0 * bbox_wid_scale
        else:
            bbox_len = float(TRAILER_TYPES[trailer_type]["len"]) * bbox_len_scale
            bbox_wid = float(TRAILER_TYPES[trailer_type]["width"]) * bbox_wid_scale

    # ============================
    # 3) Model
    # ============================
    model = build_model(model_cfg).to(device)
    bev_pretrained = train_cfg.get("bev_pretrained", None)
    bev_freeze_epochs = int(train_cfg.get("bev_freeze_epochs", 0) or 0)
    if bev_pretrained:
        missing, unexpected = load_bev_pretrained(model, bev_pretrained)
        print(f"[INFO] Loaded BEV pretrained: {bev_pretrained}")
        if missing:
            print(f"[INFO] BEV pretrained missing keys: {len(missing)}")
        if unexpected:
            print(f"[INFO] BEV pretrained unexpected keys: {len(unexpected)}")
        if bev_freeze_epochs > 0 and hasattr(model, "bev_enc"):
            for p in model.bev_enc.parameters():
                p.requires_grad = False
            print(f"[INFO] BEV encoder frozen for {bev_freeze_epochs} epoch(s)")

    lr = train_cfg.get("lr", 1e-3)
    weight_decay = train_cfg.get("weight_decay", 1e-4)
    epochs = train_cfg.get("epochs", 50)
    use_amp = train_cfg.get("amp", True)
    mem_len = int(model_cfg.get("mem_len", 0) or 0)
    mem_detach = bool(train_cfg.get("mem_detach", True))

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = build_grad_scaler(enabled=use_amp)

    # ============================
    # 4) Train loop
    # ============================
    best_val_loss = float("inf")
    centroid_debug_count = 0
    vis_saved = 0
    vis_dir = os.path.join(out_dir, "aug_vis")
    bev_x_min = None
    bev_y_min = None
    bev_res = None
    if aug_vis:
        bev_x_min = float(base_train_dataset.bev_x_range[0])
        bev_y_min = float(base_train_dataset.bev_y_range[0])
        bev_res = float(base_train_dataset.bev_res)
    if aug_vis:
        os.makedirs(vis_dir, exist_ok=True)

    for epoch in range(epochs):
        if bev_freeze_epochs > 0 and epoch == bev_freeze_epochs and hasattr(model, "bev_enc"):
            for p in model.bev_enc.parameters():
                p.requires_grad = True
            print("[INFO] BEV encoder unfrozen")
        # ============================
        # ---- TRAIN ----
        # ============================
        model.train()
        train_loss_sum = 0.0

        pbar = tqdm(train_loader, desc=f"[Train {epoch+1}/{epochs}]")
        memory_bank = []
        prev_bsz = None
        for batch in pbar:
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)

            with amp_autocast_cuda(enabled=use_amp):
                if model_cfg.get("name") == "hitch_query_transformer" and mem_len > 0:
                    if prev_bsz is not None and batch["bev"].shape[0] != prev_bsz:
                        memory_bank = []
                    prev_bsz = batch["bev"].shape[0]
                    pred, q = model(batch, memory_bank=memory_bank, return_queries=True)
                    if mem_detach:
                        q = q.detach()
                    memory_bank.append(q)
                    if len(memory_bank) > mem_len:
                        memory_bank.pop(0)
                else:
                    pred = model(batch)         # (B,2)
                gt = batch["gt"]            # (B,2)
                if centroid_debug and centroid_debug_count < centroid_debug_max:
                    centroid_xy = batch.get("centroid_xy")
                    if centroid_xy is not None:
                        theta0 = torch.atan2(-centroid_xy[:, 1], -centroid_xy[:, 0])
                        gt_theta = torch.atan2(gt[:, 1], gt[:, 0])
                        c_deg = float(theta0[0].item() * 180.0 / np.pi)
                        g_deg = float(gt_theta[0].item() * 180.0 / np.pi)
                        print(f"[DEBUG] centroid_deg={c_deg:.2f} gt_deg={g_deg:.2f}")
                        centroid_debug_count += 1
                loss = hitch_loss(
                    pred, gt,
                    alpha=loss_alpha,
                    beta=loss_beta,
                    weight_factor=weight_factor,
                    huber_delta_deg=huber_delta_deg,
                    bins=bins,
                    bin_counts=bin_counts,
                    bin_alpha=bin_alpha,
                    bin_clamp=train_cfg.get("bin_clamp", 8.0),
                    pcd=batch.get("pcd"),
                    pcd_mask=batch.get("pcd_mask"),
                    line_loss_w=line_loss_w,
                    bbox_loss_w=bbox_loss_w,
                    bbox_len=bbox_len,
                    bbox_wid=bbox_wid,
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            if aug_vis and vis_saved < aug_vis_max:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                aug_deg = batch.get("aug_rot_deg")
                if aug_deg is not None:
                    aug_deg = aug_deg.detach().cpu().numpy()
                gt_orig = batch.get("gt_orig")
                gt_aug = batch.get("gt")
                if gt_orig is not None:
                    gt_orig = gt_orig.detach().cpu().numpy()
                if gt_aug is not None:
                    gt_aug = gt_aug.detach().cpu().numpy()
                centroid_xy = batch.get("centroid_xy")
                if centroid_xy is not None:
                    centroid_xy = centroid_xy.detach().cpu().numpy()
                joint_shift_x = batch.get("joint_shift_x")
                if joint_shift_x is not None:
                    joint_shift_x = joint_shift_x.detach().cpu().numpy()
                occ_box = batch.get("occ_box")
                if occ_box is not None:
                    occ_box = occ_box.detach().cpu().numpy()
                occ_applied = batch.get("occ_applied")
                if occ_applied is not None:
                    occ_applied = occ_applied.detach().cpu().numpy()
                occ_rear_count = batch.get("occ_rear_count")
                if occ_rear_count is not None:
                    occ_rear_count = occ_rear_count.detach().cpu().numpy()
                for i in range(batch["bev"].shape[0]):
                    if vis_saved >= aug_vis_max:
                        break
                    has_occ = occ_box is not None or (occ_applied is not None and bool(occ_applied[i]))
                    if aug_deg is not None and float(aug_deg[i]) == 0.0 and not has_occ:
                        continue
                    if "bev_orig" not in batch and not has_occ:
                        continue
                    gt_o = 0.0
                    gt_a = 0.0
                    if gt_orig is not None:
                        gt_o = float(np.rad2deg(np.arctan2(gt_orig[i][1], gt_orig[i][0])))
                    if gt_aug is not None:
                        gt_a = float(np.rad2deg(np.arctan2(gt_aug[i][1], gt_aug[i][0])))
                    if "bev_orig" in batch:
                        bev_orig = batch["bev_orig"][i, 0].detach().cpu().numpy()
                        orig_title = f"orig GT={gt_o:.1f}°"
                    else:
                        bev_orig = batch["bev"][i, 0].detach().cpu().numpy()
                        orig_title = f"orig GT={gt_o:.1f}° (no orig)"
                    bev_aug = batch["bev"][i, 0].detach().cpu().numpy()
                    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
                    axes[0].imshow(bev_orig, origin="lower")
                    axes[0].set_title(orig_title)
                    axes[1].imshow(bev_aug, origin="lower")
                    axes[1].set_title(f"aug GT={gt_a:.1f}° (rot={float(aug_deg[i]):.1f}°)")
                    if centroid_xy is not None:
                        shift_x = float(joint_shift_x[i]) if joint_shift_x is not None else 0.0
                        cx = float(centroid_xy[i][0]) + shift_x
                        cy = float(centroid_xy[i][1])
                        ix = int(round((cx - bev_x_min) / bev_res))
                        iy = int(round((cy - bev_y_min) / bev_res))
                        if 0 <= ix < bev_aug.shape[0] and 0 <= iy < bev_aug.shape[1]:
                            axes[1].scatter(iy, ix, s=20, c="cyan", marker="x")
                    if occ_box is not None:
                        cx, cy, hx, hy = occ_box[i].tolist()
                        x0 = int(round((cx - hx - bev_x_min) / bev_res))
                        x1 = int(round((cx + hx - bev_x_min) / bev_res))
                        y0 = int(round((cy - hy - bev_y_min) / bev_res))
                        y1 = int(round((cy + hy - bev_y_min) / bev_res))
                        x0 = max(0, min(bev_aug.shape[0] - 1, x0))
                        x1 = max(0, min(bev_aug.shape[0] - 1, x1))
                        y0 = max(0, min(bev_aug.shape[1] - 1, y0))
                        y1 = max(0, min(bev_aug.shape[1] - 1, y1))
                        axes[1].plot([y0, y1, y1, y0, y0], [x0, x0, x1, x1, x0], color="yellow", linewidth=2)
                        cy_pix = int(round((cy - bev_y_min) / bev_res))
                        cx_pix = int(round((cx - bev_x_min) / bev_res))
                        if 0 <= cx_pix < bev_aug.shape[0] and 0 <= cy_pix < bev_aug.shape[1]:
                            axes[1].scatter(cy_pix, cx_pix, s=25, c="red", marker="+")
                        if vis_saved < 2:
                            print(f"[DEBUG] occ_box[{vis_saved}] cx={cx:.3f}, cy={cy:.3f}, hx={hx:.3f}, hy={hy:.3f}")
                    if vis_saved < 2 and occ_applied is not None:
                        print(f"[DEBUG] occ_applied={bool(occ_applied[i])} rear_count={int(occ_rear_count[i]) if occ_rear_count is not None else -1}")
                    for ax in axes:
                        ax.axis("off")
                    plt.tight_layout()
                    plt.savefig(os.path.join(vis_dir, f"bev_pair_{vis_saved:03d}.png"), dpi=150)
                    plt.close()
                    vis_saved += 1

        train_loss = train_loss_sum / len(train_loader)
        print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.6f}")

        # ============================
        # ---- VALIDATION ----
        # ============================
        model.eval()
        val_loss_sum = 0.0
        angle_errs = []
        infer_times = []
        all_theta_pred = []
        all_theta_true = []
        memory_bank = []
        prev_bsz = None

        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"[Val {epoch+1}/{epochs}]")
            for batch in pbar:
                batch = move_batch_to_device(batch, device)

                start_t = time.time()
                with amp_autocast_cuda(enabled=use_amp, dtype=torch.bfloat16):
                    if model_cfg.get("name") == "hitch_query_transformer" and mem_len > 0:
                        if prev_bsz is not None and batch["bev"].shape[0] != prev_bsz:
                            memory_bank = []
                        prev_bsz = batch["bev"].shape[0]
                        pred, q = model(batch, memory_bank=memory_bank, return_queries=True)
                        if mem_detach:
                            q = q.detach()
                        memory_bank.append(q)
                        if len(memory_bank) > mem_len:
                            memory_bank.pop(0)
                    else:
                        pred = model(batch)
                    gt = batch["gt"]
                    loss = hitch_loss(
                        pred, gt,
                        alpha=loss_alpha,
                        beta=loss_beta,
                        weight_factor=weight_factor,
                        huber_delta_deg=huber_delta_deg,
                        bins=bins,
                        bin_counts=bin_counts,
                        bin_alpha=bin_alpha,
                        bin_clamp=train_cfg.get("bin_clamp", 8.0),
                        pcd=batch.get("pcd"),
                        pcd_mask=batch.get("pcd_mask"),
                        line_loss_w=line_loss_w,
                        bbox_loss_w=bbox_loss_w,
                        bbox_len=bbox_len,
                        bbox_wid=bbox_wid,
                    )
                infer_times.append(time.time() - start_t)

                val_loss_sum += loss.item()
                pbar.set_postfix({"val_loss": f"{loss.item():.4f}"})

                # angle error accumulation
                # pred / gt 는 (cos, sin)
                cos_p, sin_p = pred[:, 0], pred[:, 1]
                cos_g, sin_g = gt[:, 0], gt[:, 1]

                theta_p = torch.atan2(sin_p, cos_p)  # [B], rad
                theta_g = torch.atan2(sin_g, cos_g)  # [B], rad

                err_rad = wrap_rad_torch(theta_p - theta_g)
                err_deg = err_rad * 180.0 / np.pi

                angle_errs.append(err_deg.cpu())
                all_theta_pred.append(theta_p.cpu())
                all_theta_true.append(theta_g.cpu())

        val_loss = val_loss_sum / len(val_loader)
        angle_errs = torch.cat(angle_errs)          # (N,)
        theta_pred = torch.cat(all_theta_pred)      # (N,) rad
        theta_true = torch.cat(all_theta_true)      # (N,) rad

        # Metrics (deg)
        rmse = torch.sqrt(torch.mean(angle_errs ** 2)).item()
        mae = torch.mean(torch.abs(angle_errs)).item()

        # p95, p99 (deg)
        abs_err = torch.abs(angle_errs).numpy()
        p95 = float(np.percentile(abs_err, 95))
        p99 = float(np.percentile(abs_err, 99))

        # R² (rad 기준)
        ss_res = torch.sum((theta_true - theta_pred) ** 2).item()
        mean_true = torch.mean(theta_true).item()
        ss_tot = torch.sum((theta_true - mean_true) ** 2).item()
        R2 = 1.0 - ss_res / (ss_tot + 1e-12)

        # inference time
        avg_infer_ms = 1000.0 * np.mean(infer_times)
        fps = 1000.0 / avg_infer_ms if avg_infer_ms > 0 else 0.0

        print(
            f"[Epoch {epoch+1}] "
            f"Val Loss={val_loss:.6f} | RMSE={rmse:.3f}° | MAE={mae:.3f}° | "
            f"R²={R2:.3f} | p95={p95:.2f}° | p99={p99:.2f}° | "
            f"Infer={avg_infer_ms:.2f}ms ({fps:.1f} FPS)"
        )

        scheduler.step()

        # ============================
        # Save last ckpt
        # ============================
        ckpt = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "opt": optimizer.state_dict(),
            "sched": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "config": cfg,
        }
        torch.save(ckpt, os.path.join(out_dir, "last.pth"))

        # ============================
        # Save best checkpoint + log
        # ============================
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(out_dir, "best.pth")
            torch.save(ckpt, best_path)

            # ---- Best metrics dict ----
            best_info = {
                "epoch": epoch + 1,
                "val_loss": float(val_loss),
                "RMSE_deg": float(rmse),
                "MAE_deg": float(mae),
                "R2": float(R2),
                "p95_deg": float(p95),
                "p99_deg": float(p99),
                "infer_ms": float(avg_infer_ms),
                "fps": float(fps),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "config": {
                    "exp_name": exp_name,
                    "batch_size": batch_size,
                    "temporal_window": temporal_window,
                    "micro_seq_length": micro_seq_length,
                    "lr": lr,
                    "epochs": epochs,
                    "model": model_cfg,
                },
                "checkpoint_path": best_path,
            }

            # ---- 1) JSON (best only) ----
            json_path = os.path.join(out_dir, "best_metrics.json")
            with open(json_path, "w") as f:
                json.dump(best_info, f, indent=2)

            # ---- 2) CSV (append) ----
            csv_path = os.path.join(out_dir, "metrics_log.csv")
            file_exists = os.path.isfile(csv_path)
            with open(csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(best_info.keys()))
                if not file_exists:
                    writer.writeheader()
                writer.writerow(best_info)

            print(f"[INFO] New best saved to {best_path}")
            print(f"[INFO] Best metrics saved → {json_path}")
            print(f"[INFO] Metrics appended → {csv_path}")

    print("[INFO] Training finished.")


if __name__ == "__main__":
    main()
