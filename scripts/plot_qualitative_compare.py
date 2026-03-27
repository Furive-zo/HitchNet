#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
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
from utils.load_dataset import HitchDataset, TRAILER_TYPES
from utils.collate import collate_fn
from utils.angle import wrap_rad_torch
from models import build_model


def parse_args():
    parser = argparse.ArgumentParser(description="Qualitative comparison figure (3x4)")
    parser.add_argument("--naive_config", type=str, default="configs/experiments/dummy_bev_resnet_regression.yaml")
    parser.add_argument("--naive_ckpt", type=str, default=None)
    parser.add_argument("--align_config", type=str, default="configs/experiments/dummy_bev_resnet_regression_norm.yaml")
    parser.add_argument("--align_ckpt", type=str, default=None)
    parser.add_argument("--proposed_config", type=str, default="configs/experiments/dummy_bev_resnet_regression_norm_aug.yaml")
    parser.add_argument("--proposed_ckpt", type=str, default=None)
    parser.add_argument("--eval_trailer_type", type=str, choices=["charger", "dummy", "temporary"], default="charger")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--num_examples", type=int, default=3)
    parser.add_argument("--fig_w", type=float, default=12.0)
    parser.add_argument("--fig_h", type=float, default=7.8)
    parser.add_argument("--panel_color", type=str, default="#e8e8e8")
    parser.add_argument("--fig_color", type=str, default="#ffffff")
    parser.add_argument("--border_color", type=str, default="#3f3f3f")
    parser.add_argument("--border_width", type=float, default=1.0)
    parser.add_argument(
        "--prefer_seq_keywords",
        type=str,
        default="scurve,reverse",
        help="Comma-separated keywords to prioritize scenarios (e.g., scurve,reverse).",
    )
    parser.add_argument(
        "--min_angle_gap_deg",
        type=float,
        default=12.0,
        help="Minimum GT-angle gap between selected examples for diversity.",
    )
    parser.add_argument(
        "--target_angles_deg",
        type=str,
        default="-20,5,40",
        help="Comma-separated GT angle targets for sample selection (e.g., -20,5,40).",
    )
    parser.add_argument(
        "--target_angle_tol_deg",
        type=float,
        default=3.0,
        help="Preferred tolerance around each target angle. Falls back to nearest if none.",
    )
    parser.add_argument("--out", type=str, default="results/qualitative_compare.png")
    return parser.parse_args()


def move_batch_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
    return out


def box_corners(theta, length, width):
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
    ix = (corners[:, 0] - x_min) / res
    iy = (corners[:, 1] - y_min) / res
    return np.stack([iy, ix], axis=1)


def build_dataset_for_model(cfg_path, eval_trailer_type, batch_size_override, num_workers_override):
    cfg = load_config(cfg_path)
    cfg_dir = os.path.dirname(cfg_path)

    ds_rel = f"datasets/{eval_trailer_type}.yaml"
    ds_path = os.path.normpath(os.path.join(cfg_dir, "..", ds_rel))
    with open(ds_path, "r") as f:
        ds_cfg = yaml.safe_load(f)
    cfg["dataset"] = ds_cfg.get("dataset", {})

    exp_cfg = cfg.get("experiment", {})
    dset_cfg = cfg["dataset"]
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

    exp_name = exp_cfg.get("name", Path(cfg_path).stem)
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

    batch_size = batch_size_override if batch_size_override is not None else train_cfg.get("batch_size", 8)
    num_workers = num_workers_override if num_workers_override is not None else dset_cfg.get("num_workers", 4)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    return cfg, dataset, loader, exp_name


def evaluate_model(cfg_path, ckpt_override, eval_trailer_type, device, batch_size, num_workers):
    cfg, dataset, loader, exp_name = build_dataset_for_model(
        cfg_path, eval_trailer_type, batch_size, num_workers
    )
    model_cfg = cfg["model"]
    ckpt_path = ckpt_override or os.path.join("ckpts", exp_name, "best.pth")

    model = build_model(model_cfg).to(device)
    if model_cfg.get("name") not in ("rule_based", "rule_based_pca", "rule_based_ols", "rule_based_mle"):
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt, strict=True)
    model.eval()

    pred_deg = []
    gt_deg = []
    err_deg = []
    frame_names = []
    idx_global = 0
    with torch.no_grad():
        for batch in loader:
            bsz = batch["gt"].shape[0]
            batch = move_batch_to_device(batch, device)
            pred = model(batch)
            gt = batch["gt"]

            theta_p = torch.atan2(pred[:, 1], pred[:, 0])
            theta_g = torch.atan2(gt[:, 1], gt[:, 0])
            e_deg = (wrap_rad_torch(theta_p - theta_g) * 180.0 / np.pi).detach().cpu().numpy()
            pd = (theta_p * 180.0 / np.pi).detach().cpu().numpy()
            gd = (theta_g * 180.0 / np.pi).detach().cpu().numpy()

            pred_deg.extend(pd.tolist())
            gt_deg.extend(gd.tolist())
            err_deg.extend(e_deg.tolist())
            for i in range(bsz):
                frame_names.append(dataset.frame_dirs[idx_global + i])
            idx_global += bsz

    pred_deg = np.array(pred_deg, dtype=np.float32)
    gt_deg = np.array(gt_deg, dtype=np.float32)
    err_deg = np.array(err_deg, dtype=np.float32)
    return {
        "cfg": cfg,
        "dataset": dataset,
        "ckpt": ckpt_path,
        "pred_deg": pred_deg,
        "gt_deg": gt_deg,
        "err_deg": err_deg,
        "frame_names": frame_names,
    }


def pick_interesting_indices(
    naive_err, align_err, proposed_err, gt_deg, frame_names, k, prefer_keywords, min_angle_gap_deg
):
    # Prefer difficult + large inter-model gap cases
    e1 = np.abs(naive_err)
    e2 = np.abs(align_err)
    e3 = np.abs(proposed_err)
    spread = np.maximum(np.maximum(e1, e2), e3) - np.minimum(np.minimum(e1, e2), e3)
    score = e1 + e2 + e3 + 2.0 * spread
    idx = np.argsort(-score)

    prefer_set = []
    for kw in prefer_keywords:
        s = kw.strip().lower()
        if s:
            prefer_set.append(s)

    preferred = []
    others = []
    for i in idx.tolist():
        seq_name = Path(frame_names[i]).parent.name.lower()
        if any(kw in seq_name for kw in prefer_set):
            preferred.append(i)
        else:
            others.append(i)
    merged = preferred + others

    selected = []
    for i in merged:
        if len(selected) >= k:
            break
        ang_i = float(gt_deg[i])
        ok = True
        for j in selected:
            if abs(ang_i - float(gt_deg[j])) < float(min_angle_gap_deg):
                ok = False
                break
        if ok:
            selected.append(i)

    # Fallback: if diversity constraint is too strict, fill remaining by score order
    if len(selected) < k:
        used = set(selected)
        for i in merged:
            if i in used:
                continue
            selected.append(i)
            if len(selected) >= k:
                break

    return selected[:k]


def pick_target_angle_indices(gt_deg, proposed_err, target_angles_deg, target_tol_deg):
    gt = np.asarray(gt_deg, dtype=np.float32)
    err = np.abs(np.asarray(proposed_err, dtype=np.float32))
    chosen = []
    used = set()

    for t in target_angles_deg:
        d = np.abs(gt - float(t))
        cand = np.where(d <= float(target_tol_deg))[0]
        if len(cand) > 0:
            # Prefer larger error inside tolerance window.
            order = sorted(cand.tolist(), key=lambda i: (err[i], -d[i]), reverse=True)
        else:
            # Fallback: nearest GT angle, tie-break by larger error.
            order = sorted(range(len(gt)), key=lambda i: (d[i], -err[i]))

        pick = None
        for i in order:
            if i not in used:
                pick = i
                break
        if pick is None:
            continue
        chosen.append(int(pick))
        used.add(int(pick))

    return chosen


def plot_cell(
    ax, bev, gt_deg, pr_deg, err_deg, dataset, trailer_type, normalize_xy,
    row_title=None, col_title=None, panel_color="#e8e8e8",
    border_color="#3f3f3f", border_width=1.0
):
    ax.set_facecolor(panel_color)
    bev_show = np.asarray(bev, dtype=np.float32)
    valid = bev_show > 0.0
    bev_masked = np.ma.masked_where(~valid, bev_show)
    vmax = float(np.percentile(bev_show[valid], 99.0)) if np.any(valid) else 1.0
    ax.imshow(bev_masked, origin="lower", cmap="gray_r", vmin=0.0, vmax=max(vmax, 1e-6))
    x_min, y_min = dataset.bev_x_range[0], dataset.bev_y_range[0]
    res = dataset.bev_res

    trailer_len = TRAILER_TYPES[trailer_type]["len"]
    trailer_width = TRAILER_TYPES[trailer_type]["width"]
    normalize_ratio = max(trailer_len, trailer_width) / 2.0
    box_len = (trailer_len / normalize_ratio if normalize_xy else trailer_len) * 1.4
    box_wid = trailer_width / normalize_ratio if normalize_xy else trailer_width

    gt_box = corners_to_pixels(box_corners(np.deg2rad(gt_deg), box_len, box_wid), x_min, y_min, res)
    pr_box = corners_to_pixels(box_corners(np.deg2rad(pr_deg), box_len, box_wid), x_min, y_min, res)
    ax.add_patch(plt.Polygon(gt_box, fill=False, edgecolor="lime", linewidth=1.3))
    ax.add_patch(plt.Polygon(pr_box, fill=False, edgecolor="red", linewidth=1.3))

    txt = f"GT {gt_deg:.1f}°\nPred {pr_deg:.1f}°\nErr {err_deg:+.1f}°"
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, ha="left", va="top",
            fontsize=12, color="white", bbox=dict(facecolor="black", alpha=0.35, pad=2, edgecolor="none"))
    if col_title is not None:
        ax.set_title(col_title, fontsize=14)
    if row_title is not None:
        ax.text(-0.10, 0.5, row_title, transform=ax.transAxes, rotation=90,
                ha="center", va="center", fontsize=16, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(border_color)
        spine.set_linewidth(border_width)


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    naive = evaluate_model(
        cfg_path=args.naive_config,
        ckpt_override=args.naive_ckpt,
        eval_trailer_type=args.eval_trailer_type,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    align = evaluate_model(
        cfg_path=args.align_config,
        ckpt_override=args.align_ckpt,
        eval_trailer_type=args.eval_trailer_type,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    proposed = evaluate_model(
        cfg_path=args.proposed_config,
        ckpt_override=args.proposed_ckpt,
        eval_trailer_type=args.eval_trailer_type,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    if len(naive["frame_names"]) != len(align["frame_names"]) or len(naive["frame_names"]) != len(proposed["frame_names"]):
        raise RuntimeError("Model sample count mismatch.")
    for i in range(len(naive["frame_names"])):
        n = Path(naive["frame_names"][i]).name
        a = Path(align["frame_names"][i]).name
        p = Path(proposed["frame_names"][i]).name
        if n != a or n != p:
            raise RuntimeError("Model frame order mismatch.")

    target_angles = []
    for s in args.target_angles_deg.split(","):
        s = s.strip()
        if s:
            target_angles.append(float(s))
    if len(target_angles) == 0:
        target_angles = [-20.0, 5.0, 40.0]
    target_angles = target_angles[:args.num_examples]

    pick_idx = pick_target_angle_indices(
        naive["gt_deg"], proposed["err_deg"], target_angles, args.target_angle_tol_deg
    )
    if len(pick_idx) < args.num_examples:
        prefer_keywords = [x for x in args.prefer_seq_keywords.split(",")]
        fallback = pick_interesting_indices(
            naive["err_deg"], align["err_deg"], proposed["err_deg"],
            naive["gt_deg"], naive["frame_names"], args.num_examples, prefer_keywords, args.min_angle_gap_deg
        )
        used = set(pick_idx)
        for i in fallback:
            if i not in used:
                pick_idx.append(i)
                used.add(i)
            if len(pick_idx) >= args.num_examples:
                break

    fig, axes = plt.subplots(3, args.num_examples, figsize=(args.fig_w, args.fig_h))
    fig.patch.set_facecolor(args.fig_color)
    if args.num_examples == 1:
        axes = np.array(axes).reshape(3, 1)

    for j, idx in enumerate(pick_idx):
        n_item = naive["dataset"][idx]
        a_item = align["dataset"][idx]
        p_item = proposed["dataset"][idx]
        n_bev = n_item["bev"][0].numpy()
        a_bev = a_item["bev"][0].numpy()
        p_bev = p_item["bev"][0].numpy()

        col_name = Path(naive["frame_names"][idx]).parent.name + "/" + Path(naive["frame_names"][idx]).name

        plot_cell(
            axes[0, j],
            bev=n_bev,
            gt_deg=float(naive["gt_deg"][idx]),
            pr_deg=float(naive["pred_deg"][idx]),
            err_deg=float(naive["err_deg"][idx]),
            dataset=naive["dataset"],
            trailer_type=args.eval_trailer_type,
            normalize_xy=bool(naive["cfg"].get("train", {}).get("normalize_xy", False)),
            row_title="Naive BEV" if j == 0 else None,
            col_title=col_name,
            panel_color=args.panel_color,
            border_color=args.border_color,
            border_width=args.border_width,
        )
        plot_cell(
            axes[1, j],
            bev=a_bev,
            gt_deg=float(align["gt_deg"][idx]),
            pr_deg=float(align["pred_deg"][idx]),
            err_deg=float(align["err_deg"][idx]),
            dataset=align["dataset"],
            trailer_type=args.eval_trailer_type,
            normalize_xy=bool(align["cfg"].get("train", {}).get("normalize_xy", False)),
            row_title="Aligned BEV" if j == 0 else None,
            col_title=None,
            panel_color=args.panel_color,
            border_color=args.border_color,
            border_width=args.border_width,
        )
        plot_cell(
            axes[2, j],
            bev=p_bev,
            gt_deg=float(proposed["gt_deg"][idx]),
            pr_deg=float(proposed["pred_deg"][idx]),
            err_deg=float(proposed["err_deg"][idx]),
            dataset=proposed["dataset"],
            trailer_type=args.eval_trailer_type,
            normalize_xy=bool(proposed["cfg"].get("train", {}).get("normalize_xy", False)),
            row_title="Proposed" if j == 0 else None,
            col_title=None,
            panel_color=args.panel_color,
            border_color=args.border_color,
            border_width=args.border_width,
        )

    # Keep panel width while tightening row gaps.
    fig.subplots_adjust(left=0.07, right=0.995, top=0.97, bottom=0.04, hspace=0.01, wspace=0.08)
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(args.out, dpi=220, facecolor=args.fig_color)
    plt.close(fig)

    print(f"[INFO] Saved figure: {args.out}")
    print(f"[INFO] Naive ckpt: {naive['ckpt']}")
    print(f"[INFO] Align ckpt: {align['ckpt']}")
    print(f"[INFO] Proposed ckpt: {proposed['ckpt']}")
    print(f"[INFO] Selected indices: {pick_idx}")


if __name__ == "__main__":
    main()
