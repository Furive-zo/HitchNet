#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Prepare per-sequence proxy-evaluation CSV logs from model inference outputs."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from models import build_model
from utils.collate import collate_fn
from utils.load_config import load_config
from utils.load_dataset import HitchDataset


R_FLU_FROM_RT_VEHICLE = np.array(
    [[0.0, 0.0, -1.0], [0.0, -1.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float64
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create CSV logs for curvature-conditioned proxy evaluation.")
    p.add_argument(
        "--models",
        type=str,
        nargs="+",
        required=True,
        help="Model specs: name|config|ckpt. ckpt can be '-' for auto best.pth",
    )
    p.add_argument("--eval_trailer_type", type=str, choices=["charger", "dummy", "temporary"], default="charger")
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--out_dir", type=str, default="logs_proxy")
    return p.parse_args()


def parse_models(specs: List[str]) -> List[Tuple[str, str, str]]:
    out = []
    for s in specs:
        parts = [x.strip() for x in s.split("|")]
        if len(parts) != 3:
            raise ValueError(f"Invalid model spec: {s}. Use name|config|ckpt")
        out.append((parts[0], parts[1], parts[2]))
    return out


def resolve_eval_dataset(cfg_path: str, eval_trailer_type: str) -> Dict:
    cfg_dir = os.path.dirname(cfg_path)
    ds_rel = f"datasets/{eval_trailer_type}.yaml"
    ds_path = os.path.normpath(os.path.join(cfg_dir, "..", ds_rel))
    with open(ds_path, "r") as f:
        ds_cfg = yaml.safe_load(f)
    return ds_cfg.get("dataset", {})


def build_dataset_loader(cfg: Dict, cfg_path: str, eval_trailer_type: str, split: str, batch_size: int | None, num_workers: int | None):
    dset_cfg = resolve_eval_dataset(cfg_path, eval_trailer_type)
    train_cfg = cfg.get("train", {})
    dataset = HitchDataset(
        root=dset_cfg["root"],
        split_json=dset_cfg["split"],
        split=split,
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
        stats_path = train_cfg.get("dlog_stats_path", os.path.join("ckpts", exp_name, f"dlog_range_stats_b{bin_size}.pt"))
        if os.path.exists(stats_path):
            stats = torch.load(stats_path, map_location="cpu", weights_only=False)
            dataset.dlog_range_stats = stats
            dataset._build_range_bin_idx()

    bs = batch_size if batch_size is not None else int(train_cfg.get("batch_size", 8))
    nw = num_workers if num_workers is not None else int(dset_cfg.get("num_workers", 4))
    loader = DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True, collate_fn=collate_fn)
    return dataset, loader, exp_name


def move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
    return out


def infer_angles(model_name: str, cfg_path: str, ckpt_spec: str, eval_trailer_type: str, split: str, device: torch.device, batch_size: int | None, num_workers: int | None):
    cfg = load_config(cfg_path)
    dataset, loader, exp_name = build_dataset_loader(cfg, cfg_path, eval_trailer_type, split, batch_size, num_workers)
    ckpt_path = os.path.join("ckpts", exp_name, "best.pth") if ckpt_spec == "-" else ckpt_spec

    model_cfg = cfg["model"]
    model = build_model(model_cfg).to(device)
    if model_cfg.get("name") not in ("rule_based", "rule_based_pca", "rule_based_ols", "rule_based_mle"):
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"[{model_name}] checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt, strict=True)
    model.eval()

    pred_deg, gt_deg, frame_dirs = [], [], []
    idx_global = 0
    with torch.no_grad():
        for batch in loader:
            bsz = batch["gt"].shape[0]
            batch = move_batch_to_device(batch, device)
            pred = model(batch)
            gt = batch["gt"]

            theta_p = torch.atan2(pred[:, 1], pred[:, 0]) * 180.0 / np.pi
            theta_g = torch.atan2(gt[:, 1], gt[:, 0]) * 180.0 / np.pi
            pred_deg.extend(theta_p.detach().cpu().numpy().tolist())
            gt_deg.extend(theta_g.detach().cpu().numpy().tolist())
            for i in range(bsz):
                frame_dirs.append(dataset.frame_dirs[idx_global + i])
            idx_global += bsz

    return np.asarray(pred_deg, dtype=np.float64), np.asarray(gt_deg, dtype=np.float64), frame_dirs


def _safe_median(vals: List[float], default: float = np.nan) -> float:
    if not vals:
        return float(default)
    arr = np.asarray(vals, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float(default)
    return float(np.median(arr))


def frame_meta(frame_dir: str, cache: Dict[str, Dict]) -> Dict:
    if frame_dir in cache:
        return cache[frame_dir]

    gt_path = os.path.join(frame_dir, "gt_hitch_angle.json")
    vel_path = os.path.join(frame_dir, "vehicle_velocity.json")
    imu_path = os.path.join(frame_dir, "vehicle_imu.json")

    if not (os.path.exists(gt_path) and os.path.exists(vel_path) and os.path.exists(imu_path)):
        raise FileNotFoundError(f"Missing one of gt/velocity/imu JSON in {frame_dir}")

    gt = json.load(open(gt_path, "r"))
    vel = json.load(open(vel_path, "r"))
    imu = json.load(open(imu_path, "r"))

    t = gt.get("odom_vehicle_stamp", np.nan)
    if not np.isfinite(t):
        t = _safe_median(vel.get("stamp_sec", []), np.nan)

    v = _safe_median(vel.get("longitudinal_velocity", []), np.nan)
    wx = _safe_median(imu.get("angular_velocity_x", []), np.nan)
    wy = _safe_median(imu.get("angular_velocity_y", []), np.nan)
    wz = _safe_median(imu.get("angular_velocity_z", []), np.nan)
    w_flu = R_FLU_FROM_RT_VEHICLE @ np.array([wx, wy, wz], dtype=np.float64)
    yaw_rate = float(w_flu[2])

    out = {"t": float(t), "v": float(v), "yaw_rate": yaw_rate}
    cache[frame_dir] = out
    return out


def main() -> None:
    args = parse_args()
    specs = parse_models(args.models)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    meta_cache: Dict[str, Dict] = {}
    for model_name, cfg_path, ckpt in specs:
        pred_deg, gt_deg, frame_dirs = infer_angles(
            model_name=model_name,
            cfg_path=cfg_path,
            ckpt_spec=ckpt,
            eval_trailer_type=args.eval_trailer_type,
            split=args.split,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        if len(frame_dirs) == 0:
            print(f"[WARN] {model_name}: no frames")
            continue

        rows_by_seq: Dict[str, List[Dict]] = {}
        for i, fr in enumerate(frame_dirs):
            seq = Path(fr).parent.name
            frame_idx = int(Path(fr).name.split("_")[-1]) if Path(fr).name.startswith("frame_") else i
            m = frame_meta(fr, meta_cache)
            row = {
                "t": m["t"],
                "v": m["v"],
                "yaw_rate": m["yaw_rate"],
                "gamma_gt": float(np.deg2rad(gt_deg[i])),
                "gamma_pred": float(np.deg2rad(pred_deg[i])),
                "sequence": seq,
                "frame_idx": frame_idx,
                "frame_dir": fr,
            }
            rows_by_seq.setdefault(seq, []).append(row)

        model_dir = os.path.join(args.out_dir, model_name.replace(" ", "_"))
        os.makedirs(model_dir, exist_ok=True)
        n_written = 0
        for seq, rows in rows_by_seq.items():
            rows.sort(key=lambda r: r["frame_idx"])
            # Enforce strictly increasing t if duplicated timestamps exist.
            t_prev = -np.inf
            for r in rows:
                if not np.isfinite(r["t"]):
                    r["t"] = t_prev + 0.1 if np.isfinite(t_prev) else 0.0
                if r["t"] <= t_prev:
                    r["t"] = t_prev + 1e-3
                t_prev = r["t"]

            out_csv = os.path.join(model_dir, f"{seq}.csv")
            with open(out_csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["t", "v", "yaw_rate", "gamma_gt", "gamma_pred", "sequence", "frame_idx", "frame_dir"])
                w.writeheader()
                w.writerows(rows)
            n_written += 1
        print(f"[INFO] {model_name}: wrote {n_written} sequence CSVs -> {model_dir}")


if __name__ == "__main__":
    main()

