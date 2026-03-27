#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from pathlib import Path
from datetime import datetime
import time
import json
import yaml
import csv
import heapq

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.load_config import load_config
from utils.load_dataset import HitchDataset, TRAILER_TYPES
from utils.collate import collate_fn
from utils.loss import hitch_loss
from utils.angle import wrap_rad_torch

from models import build_model
import matplotlib
matplotlib.use("Agg")  # SSH/headless 환경에서 저장용
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="HitchNet evaluation script")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--save_csv", action="store_true")
    parser.add_argument("--plot", action="store_true", help="Save angle-error distribution plots (PNG).")
    parser.add_argument("--plot_bins", type=float, default=5.0, help="Bin size in degrees for binned metrics/plots.")
    parser.add_argument("--save_err_bev", action="store_true", help="Save BEV for samples with large angle error.")
    parser.add_argument("--err_bev_thresh", type=float, default=2.0, help="Error threshold (deg) for saving BEV.")
    parser.add_argument("--err_bev_max", type=int, default=200, help="Max number of BEV images to save.")
    parser.add_argument("--save_bev_samples", action="store_true", help="Save sample BEV images during test.")
    parser.add_argument("--bev_sample_max", type=int, default=50, help="Max number of BEV samples to save.")
    parser.add_argument("--save_best_bev", action="store_true", help="Save BEV with minimum absolute error.")
    parser.add_argument("--save_attn", action="store_true", help="Save hitch-query attention sampling overlay.")
    parser.add_argument("--attn_max", type=int, default=50, help="Max number of attention overlays to save.")
    parser.add_argument("--trailer_type", type=str, choices=["charger", "dummy", "temporary"], default=None,
                        help="Trailer type to select dataset config.")
    parser.add_argument("--exp_name", type=str, default=None, help="Override experiment name for output dir.")
    return parser.parse_args()


def move_batch_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
    return out


def count_parameters(model):
    """Return total number of trainable parameters (million scale)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

def add_origin_marker(ax, x_min, y_min, res):
    ix = (0.0 - x_min) / res
    iy = (0.0 - y_min) / res
    ax.scatter([iy], [ix], s=18, c="white", marker="x")
    ax.text(iy + 2, ix + 2, "O", color="white", fontsize=6)

def plot_attn_overlay(bev0, ref_points, sample_points, weights, out_path, title=None):
    h, w = bev0.shape
    plt.figure(figsize=(4, 4))
    plt.imshow(bev0, origin="lower")
    ax = plt.gca()
    nq = ref_points.shape[0]
    cmap = plt.cm.get_cmap("tab10", max(nq, 1))
    for qi in range(nq):
        color = cmap(qi)
        rx = ref_points[qi, 1] * (w - 1)
        ry = ref_points[qi, 0] * (h - 1)
        ax.scatter([rx], [ry], s=40, c=[color], marker="x")
        for k in range(sample_points.shape[1]):
            sx = sample_points[qi, k, 1] * (w - 1)
            sy = sample_points[qi, k, 0] * (h - 1)
            size = 20.0 + 120.0 * float(weights[qi, k])
            ax.scatter([sx], [sy], s=size, c=[color], alpha=0.7)
    if title:
        plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def box_corners(theta, length, width):
    # Rectangle anchored at hitch point: x in [-L, 0], y in [-W/2, W/2]
    corners = np.array(
        [[-length, -width / 2],
         [0.0, -width / 2],
         [0.0, width / 2],
         [-length, width / 2]],
        dtype=np.float32,
    )
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    return corners @ rot.T


def corners_to_pixels(corners, x_min, y_min, res):
    # corners: (4,2) in (x,y)
    ix = (corners[:, 0] - x_min) / res
    iy = (corners[:, 1] - y_min) / res
    return np.stack([iy, ix], axis=1)  # (4,2) as (col,row)

def main():
    args = parse_args()

    # ============================
    # 1) Config & Device
    # ============================
    cfg = load_config(args.config)
    cfg_dir = os.path.dirname(args.config)
    if args.trailer_type:
        ds_rel = f"datasets/{args.trailer_type}.yaml"
        ds_path = os.path.normpath(os.path.join(cfg_dir, "..", ds_rel))
        with open(ds_path, "r") as f:
            ds_cfg = yaml.safe_load(f)
        cfg["dataset"] = ds_cfg.get("dataset", {})
    exp_cfg = cfg.get("experiment", {})
    if args.exp_name:
        exp_cfg["name"] = args.exp_name
    dset_cfg = cfg["dataset"]
    model_cfg = cfg["model"]
    test_cfg = cfg.get("train", {})

    exp_name = exp_cfg.get("name", Path(args.config).stem)
    out_dir = exp_cfg.get("output_dir", os.path.join("ckpts", exp_name))
    if args.trailer_type:
        out_dir = os.path.join("results", exp_name, f"{args.trailer_type}_trailer")
    os.makedirs(out_dir, exist_ok=True)
    out_dir_plot = out_dir

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # ============================
    # 2) Dataset (test split)
    # ============================
    num_workers = args.num_workers or dset_cfg.get("num_workers", 4)
    batch_size = test_cfg.get("batch_size", cfg["train"].get("batch_size", 8))
    trailer_type = dset_cfg.get("name", "charger")
    train_cfg = cfg.get("train", {})
    normalize_xy = test_cfg.get("normalize_xy", train_cfg.get("normalize_xy", False))
    bev_add_xy = test_cfg.get("bev_add_xy", train_cfg.get("bev_add_xy", False))
    bev_add_orient = test_cfg.get("bev_add_orient", train_cfg.get("bev_add_orient", False))
    bev_use_hmax = test_cfg.get("bev_use_hmax", train_cfg.get("bev_use_hmax", True))
    bev_use_dlog = test_cfg.get("bev_use_dlog", train_cfg.get("bev_use_dlog", True))
    occ_binary = test_cfg.get("occ_binary", train_cfg.get("occ_binary", False))
    joint_shift_x = float(test_cfg.get("joint_shift_x", train_cfg.get("joint_shift_x", 0.8)))
    add_observed_mask = test_cfg.get("add_observed_mask", train_cfg.get("add_observed_mask", False))
    observed_bins = int(test_cfg.get("observed_bins", train_cfg.get("observed_bins", 360)))
    observed_margin = float(test_cfg.get("observed_margin", train_cfg.get("observed_margin", 0.0)))
    dlog_range_norm = test_cfg.get("dlog_range_norm", train_cfg.get("dlog_range_norm", False))
    dlog_range_norm_mode = test_cfg.get("dlog_range_norm_mode", train_cfg.get("dlog_range_norm_mode", "center"))
    dlog_range_bin_size = float(test_cfg.get("dlog_range_bin_size", train_cfg.get("dlog_range_bin_size", 0.1)))
    dlog_stats_path = test_cfg.get("dlog_stats_path", os.path.join("ckpts", exp_name, f"dlog_range_stats_b{dlog_range_bin_size}.pt"))
    centroid_mode = test_cfg.get("centroid_mode", train_cfg.get("centroid_mode", "minmax"))
    model_name = model_cfg.get("name")
    fast_rule_mode = model_name in ("rule_based", "rule_based_pca", "rule_based_ols", "rule_based_mle")
    need_bev_for_viz = bool(args.save_err_bev or args.save_bev_samples or args.save_best_bev or args.save_attn)
    trailer_len = TRAILER_TYPES[trailer_type]["len"]
    trailer_width = TRAILER_TYPES[trailer_type]["width"]
    normalize_ratio = max(trailer_len, trailer_width) / 2
    box_len = (trailer_len/normalize_ratio if normalize_xy else trailer_len) * 1.4
    box_wid = (trailer_width/normalize_ratio if normalize_xy else trailer_width)
    x_min, x_max = -4.0, 0.5
    y_min, y_max = -4.0, 4.0
    res = 0.033

    test_dataset = HitchDataset(
        root=dset_cfg["root"],
        split_json=dset_cfg["split"],
        split="test",
        temporal_window=dset_cfg.get("temporal_window", 20),
        micro_seq_length=dset_cfg.get("micro_seq_length", 10),
        trailer_type=trailer_type,
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
        dlog_range_norm=dlog_range_norm,
        dlog_range_norm_mode=dlog_range_norm_mode,
        joint_shift_x=joint_shift_x,
        skip_temporal=fast_rule_mode,
        build_bev=(not fast_rule_mode) or need_bev_for_viz,
    )
    if dlog_range_norm and os.path.exists(dlog_stats_path):
        stats = torch.load(dlog_stats_path, map_location="cpu", weights_only=False)
        test_dataset.dlog_range_stats = stats
        test_dataset._build_range_bin_idx()
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    print(f"[INFO] HitchDataset split=test, frames={len(test_dataset)}")

    # ============================
    # 3) Load model & checkpoint
    # ============================
    model = build_model(model_cfg).to(device)
    if model_cfg.get("name") not in ("rule_based", "rule_based_pca", "rule_based_ols", "rule_based_mle") and os.path.exists(args.ckpt):
        ckpt = torch.load(args.ckpt, map_location=device)
        # 2가지 저장 형식 대비
        model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt, strict=True)
    model.eval()

    # ============================
    # 4) Parameter Count
    # ============================
    mparams = count_parameters(model)
    print(f"[INFO] Model parameters: {mparams:.3f} M")

    # ============================
    # 5) Test loop
    # ============================
    angle_errs = []
    theta_pred_list = []
    theta_true_list = []
    infer_times = []
    saved_err_bev = 0
    saved_bev_samples = 0
    err_bev_heap = []
    err_bev_counter = 0
    attn_saved = 0
    min_err = None
    min_err_bev = None
    min_err_gt = None
    min_err_pr = None

    total_loss = 0.0
    n_batches = 0
    mem_len = int(model_cfg.get("mem_len", 0) or 0)
    mem_detach = bool(test_cfg.get("mem_detach", True))
    memory_bank = []
    prev_bsz = None

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="[Test]")
        for batch in pbar:
            batch = move_batch_to_device(batch, device)

            start_t = time.time()
            if model_cfg.get("name") == "hitch_query_transformer":
                if mem_len > 0:
                    if prev_bsz is not None and batch["bev"].shape[0] != prev_bsz:
                        memory_bank = []
                    prev_bsz = batch["bev"].shape[0]
                    if args.save_attn:
                        pred, q, attn_info = model(
                            batch, memory_bank=memory_bank, return_queries=True, return_attn=True
                        )
                    else:
                        pred, q = model(batch, memory_bank=memory_bank, return_queries=True)
                        attn_info = None
                    if mem_detach:
                        q = q.detach()
                    memory_bank.append(q)
                    if len(memory_bank) > mem_len:
                        memory_bank.pop(0)
                else:
                    if args.save_attn:
                        pred, attn_info = model(batch, return_attn=True)
                    else:
                        pred = model(batch)
                        attn_info = None
            else:
                pred = model(batch)
                attn_info = None
            infer_times.append(time.time() - start_t)

            gt = batch["gt"]
            loss = hitch_loss(pred, gt)
            total_loss += loss.item()
            n_batches += 1

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            # angle
            cos_p, sin_p = pred[:, 0], pred[:, 1]
            cos_g, sin_g = gt[:, 0], gt[:, 1]

            theta_p = torch.atan2(sin_p, cos_p)
            theta_g = torch.atan2(sin_g, cos_g)

            err_deg = wrap_rad_torch(theta_p - theta_g) * 180.0 / np.pi
            err_abs = torch.abs(err_deg)

            angle_errs.append(err_deg.cpu())
            theta_pred_list.append(theta_p.cpu())
            theta_true_list.append(theta_g.cpu())

            if args.save_attn and attn_info is not None and "bev" in batch and attn_saved < args.attn_max:
                ref_points = attn_info.get("ref_points")
                sample_points = attn_info.get("sample_points")
                weights = attn_info.get("weights")
                if ref_points is not None and sample_points is not None and weights is not None:
                    for i in range(batch["bev"].shape[0]):
                        if attn_saved >= args.attn_max:
                            break
                        bev0 = batch["bev"][i, 0].detach().cpu().numpy()
                        ref_i = ref_points[i].detach().cpu().numpy()
                        samp_i = sample_points[i].detach().cpu().numpy()
                        w_i = weights[i].detach().cpu().numpy()
                        gt_i = float(theta_g[i].item() * 180.0 / np.pi)
                        pr_i = float(theta_p[i].item() * 180.0 / np.pi)
                        fpath = os.path.join(out_dir, "attn")
                        os.makedirs(fpath, exist_ok=True)
                        title = f"GT {gt_i:.1f}°, Pred {pr_i:.1f}°"
                        out_path = os.path.join(fpath, f"attn_{attn_saved:04d}.png")
                        plot_attn_overlay(bev0, ref_i, samp_i, w_i, out_path, title=title)
                        attn_saved += 1

            if args.save_best_bev and "bev" in batch:
                err_abs_np = err_abs.detach().cpu().numpy()
                for i in range(err_abs_np.shape[0]):
                    e = float(err_abs_np[i])
                    if (min_err is None) or (e < min_err):
                        min_err = e
                        min_err_bev = batch["bev"][i, 0].detach().cpu().numpy()
                        min_err_gt = float(theta_g[i].item() * 180.0 / np.pi)
                        min_err_pr = float(theta_p[i].item() * 180.0 / np.pi)

            if args.save_err_bev and "bev" in batch and args.err_bev_max > 0:
                err_np = err_deg.detach().cpu().numpy()
                for i in range(err_np.shape[0]):
                    err_i = float(err_np[i])
                    if abs(err_i) < args.err_bev_thresh:
                        continue
                    bev0 = batch["bev"][i, 0].detach().cpu().numpy()
                    gt_i = float(theta_g[i].item() * 180.0 / np.pi)
                    pr_i = float(theta_p[i].item() * 180.0 / np.pi)
                    centroid_xy = None
                    if "centroid_xy" in batch:
                        centroid_xy = batch["centroid_xy"][i].detach().cpu().numpy()
                    item = (abs(err_i), err_bev_counter, bev0, gt_i, pr_i, err_i, centroid_xy)
                    err_bev_counter += 1
                    if len(err_bev_heap) < args.err_bev_max:
                        heapq.heappush(err_bev_heap, item)
                    else:
                        if item[0] > err_bev_heap[0][0]:
                            heapq.heapreplace(err_bev_heap, item)

            if args.save_bev_samples and "bev" in batch and saved_bev_samples < args.bev_sample_max:
                for i in range(batch["bev"].shape[0]):
                    if saved_bev_samples >= args.bev_sample_max:
                        break
                    bev0 = batch["bev"][i, 0].detach().cpu().numpy()
                    fpath = os.path.join(out_dir, "bev_samples")
                    os.makedirs(fpath, exist_ok=True)
                    plt.figure(figsize=(4, 4))
                    plt.imshow(bev0, origin="lower")
                    ax = plt.gca()
                    add_origin_marker(ax, x_min, y_min, res)
                    plt.title("BEV occ (test sample)")
                    plt.axis("off")
                    plt.tight_layout()
                    plt.savefig(os.path.join(fpath, f"bev_{saved_bev_samples:04d}.png"), dpi=150)
                    plt.close()
                    saved_bev_samples += 1

            if args.save_err_bev and err_bev_heap:
                fpath = os.path.join(out_dir, "err_bev")
                os.makedirs(fpath, exist_ok=True)
                # sort by abs error desc
                for idx, (_, _, bev0, gt_i, pr_i, err_i, centroid_xy) in enumerate(
                    sorted(err_bev_heap, key=lambda x: x[0], reverse=True)
                ):
                    fname = f"err_bev_{idx:04d}.png"
                    plt.figure(figsize=(4, 4))
                    plt.imshow(bev0, origin="lower")
                    ax = plt.gca()
                    add_origin_marker(ax, x_min, y_min, res)
                    if centroid_xy is not None:
                        cx, cy = float(centroid_xy[0]), float(centroid_xy[1])
                        if "joint_shift_x" in batch:
                            cx = cx + float(batch["joint_shift_x"][0].item())
                        cx_pix = (cx - x_min) / res
                        cy_pix = (cy - y_min) / res
                        ax.scatter([cy_pix], [cx_pix], s=35, c="cyan", marker="o", label="centroid")
                    plt.title(f"GT {gt_i:.1f}°, Pred {pr_i:.1f}°, Err {err_i:.1f}°")
                    gt_box = corners_to_pixels(
                        box_corners(np.deg2rad(gt_i), box_len, box_wid), x_min, y_min, res
                    )
                    pr_box = corners_to_pixels(
                        box_corners(np.deg2rad(pr_i), box_len, box_wid), x_min, y_min, res
                    )
                    plt.gca().add_patch(
                        plt.Polygon(gt_box, fill=False, edgecolor="lime", linewidth=1.5, label="GT")
                    )
                    plt.gca().add_patch(
                        plt.Polygon(pr_box, fill=False, edgecolor="red", linewidth=1.5, label="Pred")
                    )
                    plt.axis("off")
                    plt.tight_layout()
                    plt.savefig(os.path.join(fpath, fname), dpi=150)
                    plt.close()
    avg_loss = total_loss / max(n_batches, 1)

    # ============================
    # 6) Metrics
    # ============================
    angle_errs = torch.cat(angle_errs)
    abs_err = torch.abs(angle_errs)

    rmse = torch.sqrt(torch.mean(angle_errs ** 2)).item()
    mae = torch.mean(abs_err).item()
    max_err = abs_err.max().item()

    p95 = float(np.percentile(abs_err.numpy(), 95))
    p99 = float(np.percentile(abs_err.numpy(), 99))

    theta_pred = torch.cat(theta_pred_list)
    theta_true = torch.cat(theta_true_list)

    # R2
    ss_res = torch.sum((theta_true - theta_pred) ** 2).item()
    mean_true = torch.mean(theta_true).item()
    ss_tot = torch.sum((theta_true - mean_true) ** 2).item()
    R2 = 1.0 - ss_res / (ss_tot + 1e-12)

    # latency
    avg_infer_ms = 1000.0 * np.mean(infer_times)
    fps = 1000.0 / avg_infer_ms if avg_infer_ms > 0 else 0.0

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # ============================
    # 6.5) Angle-wise distribution plots (optional)
    # ============================
    if args.plot:
        out_dir_plot = out_dir
        os.makedirs(out_dir_plot, exist_ok=True)

        # radians -> degrees
        theta_true_deg = theta_true * 180.0 / np.pi
        theta_pred_deg = theta_pred * 180.0 / np.pi

        # signed error in deg (wrap applied already in angle_errs)
        err_deg = angle_errs.numpy()
        abs_err_deg = np.abs(err_deg)

        # ---------
        # 1) GT angle histogram
        # ---------
        plt.figure(figsize=(7, 3.5))
        plt.hist(theta_true_deg, bins=60)
        plt.xlabel("GT Hitch Angle (deg)")
        plt.ylabel("Count")
        plt.title("GT Hitch Angle Distribution")
        plt.grid(True)
        hist_path = os.path.join(out_dir_plot, f"angle_hist_{timestamp}.png")
        plt.tight_layout()
        plt.savefig(hist_path, dpi=150)
        plt.close()
        print(f"[INFO] Plot saved → {hist_path}")

        # ---------
        # 2) Scatter: abs error vs GT angle
        # ---------
        plt.figure(figsize=(7, 4))
        plt.scatter(theta_true_deg, abs_err_deg, s=4, alpha=0.25)
        plt.xlabel("GT Hitch Angle (deg)")
        plt.ylabel("Absolute Error (deg)")
        plt.title("Abs Error vs GT Hitch Angle")
        plt.ylim(0, 10)
        plt.grid(True)
        scat_path = os.path.join(out_dir_plot, f"scatter_err_vs_angle_{timestamp}.png")
        plt.tight_layout()
        plt.savefig(scat_path, dpi=150)
        plt.close()
        print(f"[INFO] Plot saved → {scat_path}")
        # ---------
        # 2.5) Scatter: Pred vs GT angle
        # ---------
        plt.figure(figsize=(5.5, 5.5))
        plt.scatter(theta_true_deg, theta_pred_deg, s=4, alpha=0.25)
        lo2, hi2 = -90.0, 90.0
        plt.plot([lo2, hi2], [lo2, hi2], linestyle="--", color="gray", linewidth=1)
        plt.xlim(lo2, hi2)
        plt.ylim(lo2, hi2)
        plt.xlabel("GT Hitch Angle (deg)")
        plt.ylabel("Pred Hitch Angle (deg)")
        plt.title("Pred vs GT Hitch Angle")
        plt.grid(True)
        pv_path = os.path.join(out_dir_plot, f"pred_vs_gt_{timestamp}.png")
        plt.tight_layout()
        plt.savefig(pv_path, dpi=150)
        plt.close()
        print(f"[INFO] Plot saved → {pv_path}")

        # ---------
        # 3) Binned MAE / p95 vs GT angle
        # ---------
        bin_size = float(args.plot_bins)
        # 범위는 데이터 기반으로 잡되, 보기 좋게 -90~90으로 클램프(원하면 바꿔도 됨)
        lo, hi = -90.0, 90.0
        bins = np.arange(lo, hi + bin_size, bin_size)
        centers = (bins[:-1] + bins[1:]) / 2.0

        mae_per_bin = []
        p95_per_bin = []
        count_per_bin = []

        for i in range(len(bins) - 1):
            m = (theta_true_deg >= bins[i]) & (theta_true_deg < bins[i + 1])
            cnt = int(m.sum())
            count_per_bin.append(cnt)
            if cnt > 0:
                mae_per_bin.append(float(np.mean(abs_err_deg[m])))
                # bin 샘플이 너무 적으면 p95가 불안정하니 조건 걸기
                if cnt >= 20:
                    p95_per_bin.append(float(np.percentile(abs_err_deg[m], 95)))
                else:
                    p95_per_bin.append(float("nan"))
            else:
                mae_per_bin.append(float("nan"))
                p95_per_bin.append(float("nan"))

        # MAE plot
        plt.figure(figsize=(7, 4))
        plt.plot(centers, mae_per_bin, marker="o")
        plt.xlabel("GT Hitch Angle (deg)")
        plt.ylabel("MAE (deg)")
        plt.title(f"MAE vs GT Hitch Angle (bin={bin_size}°)")
        plt.grid(True)
        mae_path = os.path.join(out_dir_plot, f"mae_vs_angle_{timestamp}.png")
        plt.tight_layout()
        plt.savefig(mae_path, dpi=150)
        plt.close()
        print(f"[INFO] Plot saved → {mae_path}")

        # p95 plot
        plt.figure(figsize=(7, 4))
        plt.plot(centers, p95_per_bin, marker="o")
        plt.xlabel("GT Hitch Angle (deg)")
        plt.ylabel("p95 Abs Error (deg)")
        plt.title(f"p95 Error vs GT Hitch Angle (bin={bin_size}°)")
        plt.grid(True)
        p95_path = os.path.join(out_dir_plot, f"p95_vs_angle_{timestamp}.png")
        plt.tight_layout()
        plt.savefig(p95_path, dpi=150)
        plt.close()
        print(f"[INFO] Plot saved → {p95_path}")

        # ---------
        # 4) Save binned metrics JSON (optional but useful)
        # ---------
        binned = {
            "bin_size_deg": bin_size,
            "range_deg": [lo, hi],
            "centers_deg": centers.tolist(),
            "count": count_per_bin,
            "mae_deg": mae_per_bin,
            "p95_deg": p95_per_bin,
        }
        binned_path = os.path.join(out_dir_plot, f"binned_metrics_{timestamp}.json")
        with open(binned_path, "w") as f:
            json.dump(binned, f, indent=2)
        print(f"[INFO] Binned metrics saved → {binned_path}")

    # ============================
    # 7) Print summary
    # ============================
    print("=========================================")
    print(f"[TEST] Loss={avg_loss:.6f}")
    print(f"[TEST] RMSE={rmse:.3f}° | MAE={mae:.3f}° | MaxErr={max_err:.3f}°")
    print(f"[TEST] p95={p95:.3f}° | p99={p99:.3f}° | R²={R2:.3f}")
    print(f"[TEST] Infer={avg_infer_ms:.2f} ms/step ({fps:.1f} FPS)")
    print(f"[TEST] Parameters={mparams:.3f} M")
    print("=========================================")

    if args.save_best_bev and min_err_bev is not None:
        best_dir = os.path.join(out_dir, "bev_best")
        os.makedirs(best_dir, exist_ok=True)
        plt.figure(figsize=(4, 4))
        plt.imshow(min_err_bev, origin="lower")
        ax = plt.gca()
        add_origin_marker(ax, x_min, y_min, res)
        plt.title(f"Best BEV | GT {min_err_gt:.1f}°, Pred {min_err_pr:.1f}°, Err {min_err:.2f}°")
        gt_box = corners_to_pixels(
            box_corners(np.deg2rad(min_err_gt), box_len, box_wid), x_min, y_min, res
        )
        pr_box = corners_to_pixels(
            box_corners(np.deg2rad(min_err_pr), box_len, box_wid), x_min, y_min, res
        )
        plt.gca().add_patch(
            plt.Polygon(gt_box, fill=False, edgecolor="lime", linewidth=1.5, label="GT")
        )
        plt.gca().add_patch(
            plt.Polygon(pr_box, fill=False, edgecolor="red", linewidth=1.5, label="Pred")
        )
        plt.axis("off")
        plt.tight_layout()
        best_path = os.path.join(best_dir, "bev_min_error.png")
        plt.savefig(best_path, dpi=150)
        plt.close()
        print(f"[INFO] Best-error BEV saved → {best_path}")

    # ============================
    # 8) Save metrics JSON
    # ============================
    metrics = {
        "timestamp": timestamp,
        "config": args.config,
        "checkpoint": args.ckpt,
        "loss": float(avg_loss),
        "RMSE_deg": float(rmse),
        "MAE_deg": float(mae),
        "MaxErr_deg": float(max_err),
        "p95_deg": float(p95),
        "p99_deg": float(p99),
        "R2": float(R2),
        "infer_ms": float(avg_infer_ms),
        "fps": float(fps),
        "Params_M": float(mparams),
    }

    json_path = os.path.join(out_dir_plot, f"test_metrics_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[INFO] Test metrics saved → {json_path}")

    # ============================
    # 9) Optionally save per-frame CSV
    # ============================
    if args.save_csv:
        csv_path = os.path.join(out_dir_plot, f"test_errors_{timestamp}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["theta_true_deg", "theta_pred_deg", "err_deg"])

            theta_true_deg = theta_true * 180.0 / np.pi
            theta_pred_deg = theta_pred * 180.0 / np.pi

            for tg, tp, e in zip(theta_true_deg.numpy(),
                                 theta_pred_deg.numpy(),
                                 angle_errs.numpy()):
                writer.writerow([float(tg), float(tp), float(e)])

        print(f"[INFO] Per-frame errors saved → {csv_path}")


if __name__ == "__main__":
    main()
