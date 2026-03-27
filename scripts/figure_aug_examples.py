#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse

import numpy as np
import open3d as o3d

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import yaml
import torch

from utils.load_dataset import points_to_bev, TRAILER_TYPES


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"],
    "font.size": 12,
})


def parse_args():
    parser = argparse.ArgumentParser(
        description="Augmentation examples (a) reference, (b) occlusion."
    )
    parser.add_argument("--config", type=str, default=None,
                        help="Experiment yaml (uses dataset_config + train settings).")
    parser.add_argument("--dataset_root", type=str, default="datasets/LI-HAE/dataset")
    parser.add_argument("--split_dir", type=str, default="datasets/LI-HAE/splits")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--angle_deg", type=float, required=True, help="Target hitch angle (deg).")
    parser.add_argument("--angle_tol", type=float, default=2.0)
    parser.add_argument("--max_frames", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7)

    # BEV/ROI params (match dataset defaults)
    parser.add_argument("--joint_shift_x", type=float, default=0.8)
    parser.add_argument("--res", type=float, default=0.033)
    parser.add_argument("--clip_count", type=float, default=10.0)
    parser.add_argument("--x_min", type=float, default=-2.5)
    parser.add_argument("--x_max", type=float, default=2.5)
    parser.add_argument("--y_min", type=float, default=-4.0)
    parser.add_argument("--y_max", type=float, default=0.5)

    # Alignment toggles
    parser.add_argument("--normalize_xy", action="store_true", default=True)
    parser.add_argument("--dlog_range_norm", action="store_true", default=False)
    parser.add_argument("--dlog_range_norm_mode", type=str, default="center")
    parser.add_argument("--dlog_range_bin_size", type=float, default=0.1)
    parser.add_argument("--dlog_stats_path", type=str, default=None)
    parser.add_argument("--add_observed_mask", action="store_true", default=False)
    parser.add_argument("--observed_bins", type=int, default=360)
    parser.add_argument("--observed_margin", type=float, default=0.0)
    parser.add_argument("--occ_binary", action="store_true", default=False)

    # Occlusion aug params (b)
    parser.add_argument("--occ_x_thresh", type=float, default=-2.5)
    parser.add_argument("--occ_y_thresh", type=float, default=None)
    parser.add_argument("--occ_box_x", type=float, default=1.0)
    parser.add_argument("--occ_box_y", type=float, default=1.0)
    parser.add_argument("--occ_seed", type=int, default=7)
    parser.add_argument("--trans_max_m", type=float, default=1.0)
    parser.add_argument("--trans_dir", type=str, default="forward")
    parser.add_argument("--trans_seed", type=int, default=7)
    parser.add_argument("--rot_deg", type=float, default=20.0)
    parser.add_argument("--rot_seed", type=int, default=7)
    parser.add_argument("--density_seed", type=int, default=7)
    parser.add_argument("--density_percentile", type=float, default=5.0)

    # Rendering (match figure_bev_compare)
    parser.add_argument("--panel_color", type=str, default="#e8e8e8")
    parser.add_argument("--fig_color", type=str, default="#ffffff")
    parser.add_argument("--point_size", type=float, default=3.0)
    parser.add_argument("--point_alpha", type=float, default=0.6)
    parser.add_argument("--rotate_deg", type=float, default=90.0,
                        help="Rotate points for display (deg, CCW).")
    parser.add_argument("--crop", action="store_true", default=True,
                        help="Crop points to x/y limits for display.")
    parser.add_argument("--hitch_x", type=float, default=0.0)
    parser.add_argument("--hitch_y", type=float, default=0.0)
    parser.add_argument("--show_ref_ray", action="store_true", default=True)
    parser.add_argument("--ray_len", type=float, default=1.0)
    parser.add_argument("--out", type=str, default="results/fig_aug_examples.png")
    return parser.parse_args()


def load_split_sequences(split_path, split):
    with open(split_path, "r") as f:
        data = json.load(f)
    return data.get(split, [])


def list_frames(root, seqs):
    frames = []
    for seq in seqs:
        seq_dir = os.path.join(root, seq)
        if not os.path.isdir(seq_dir):
            continue
        for d in sorted(os.listdir(seq_dir)):
            if d.startswith("frame_"):
                frames.append(os.path.join(seq_dir, d))
    return frames


def read_angle_deg(frame_dir):
    path = os.path.join(frame_dir, "gt_hitch_angle.json")
    with open(path, "r") as f:
        js = json.load(f)
    return float(js.get("gt_hitch_angle_deg", 0.0))


def read_pcd(frame_dir):
    pcd_path = os.path.join(frame_dir, "trailer_point.pcd")
    pcd = o3d.io.read_point_cloud(pcd_path)
    pts = np.asarray(pcd.points, dtype=np.float32)
    return pts


def select_frame(root, split_json, split, target_deg, tol_deg, max_frames, seed):
    seqs = load_split_sequences(split_json, split)
    frames = list_frames(root, seqs)
    rng = np.random.default_rng(seed)
    rng.shuffle(frames)
    used = 0
    for fr in frames:
        if max_frames > 0 and used >= max_frames:
            break
        angle = read_angle_deg(fr)
        used += 1
        if abs(angle - target_deg) <= tol_deg:
            return fr, angle
    return None, None


def add_panel(ax, pts_xy, panel_color, xlim, ylim, label, point_size, point_alpha, colors=None):
    ax.set_facecolor(panel_color)
    rect = plt.Rectangle(
        (xlim[0], ylim[0]),
        xlim[1] - xlim[0],
        ylim[1] - ylim[0],
        facecolor=panel_color,
        edgecolor="#000000",
        linewidth=0.6,
        zorder=0,
    )
    ax.add_patch(rect)
    if pts_xy.shape[0] > 0:
        if colors is None:
            ax.scatter(pts_xy[:, 0], pts_xy[:, 1], s=point_size, c="black", alpha=point_alpha, linewidths=0)
        else:
            ax.scatter(pts_xy[:, 0], pts_xy[:, 1], s=point_size, c=colors, linewidths=0)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal", "box")
    ax.axis("off")
    if label:
        y_off = -0.045 if label.startswith("(a)") else -0.06
        ax.text(0.5, y_off, label, transform=ax.transAxes,
                ha="center", va="top", fontsize=18, clip_on=False)


def normalize_image(img, lo=2.0, hi=98.0):
    vals = img[np.isfinite(img) & (img > 0)]
    if vals.size == 0:
        return np.zeros_like(img, dtype=np.float32)
    vmin = np.percentile(vals, lo)
    vmax = np.percentile(vals, hi)
    if vmax - vmin < 1e-6:
        return np.zeros_like(img, dtype=np.float32)
    out = (img - vmin) / (vmax - vmin)
    return np.clip(out, 0.0, 1.0)


def dlog_norm_map(pts_xy, x_range, y_range, res, clip_count, occ_binary,
                  stats_path=None, mode="center"):
    bev = points_to_bev(
        pts_xy,
        x_range=x_range,
        y_range=y_range,
        z_range=(-2.0, 2.0),
        res=res,
        clip_count=clip_count,
        joint_shift_x=0.0,
        use_hmax=False,
        use_dlog=True,
        occ_binary=occ_binary,
        add_observed_mask=False,
        observed_bins=360,
        observed_margin=0.0,
        add_xy=False,
        add_orient=False,
    )
    dlog = bev[1]
    if stats_path and os.path.exists(stats_path):
        stats = torch.load(stats_path, map_location="cpu", weights_only=False)
        bin_size = float(stats["bin_size"])
        nbins = int(stats["nbins"])
        x_min, x_max = x_range
        y_min, y_max = y_range
        H = int(np.ceil((x_max - x_min) / res))
        W = int(np.ceil((y_max - y_min) / res))
        x_centers = x_min + (np.arange(H) + 0.5) * res
        y_centers = y_min + (np.arange(W) + 0.5) * res
        x_grid, y_grid = np.meshgrid(x_centers, y_centers, indexing="ij")
        r = np.sqrt(x_grid ** 2 + y_grid ** 2)
        bin_idx = np.floor(r / bin_size).astype(np.int64)
        bin_idx = np.clip(bin_idx, 0, nbins - 1)
        mu = stats["mu"]
        sigma = stats["sigma"]
        mu_map = mu[bin_idx]
        sig_map = sigma[bin_idx]
        if mode == "zscore":
            dlog = (dlog - mu_map) / (sig_map + 1e-6)
        else:
            dlog = dlog - mu_map
    return dlog


def apply_occ_aug(pcd, gt_deg, joint_shift_x, occ_x_thresh, occ_box_x, occ_box_y, rng, occ_y_thresh=None):
    if pcd.shape[0] == 0:
        return pcd, None
    # Build box in BEV rear region, then rotate by gt around hitch
    theta = np.deg2rad(float(gt_deg))
    c, s = np.cos(-theta), np.sin(-theta)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    hitch = np.array([[-joint_shift_x, 0.0]], dtype=np.float32)
    pts_bev = (R @ (pcd[:, :2] - hitch).T).T + hitch
    x_joint = pts_bev[:, 0] + joint_shift_x
    y_joint = pts_bev[:, 1]
    if occ_y_thresh is not None:
        rear_mask = y_joint < float(occ_y_thresh)
    else:
        rear_mask = x_joint < occ_x_thresh
    if not np.any(rear_mask):
        return pcd, None
    if occ_box_x > 0.0 and occ_box_y > 0.0:
        rear_x = x_joint[rear_mask]
        rear_y = y_joint[rear_mask]
        x_min = float(rear_x.min())
        x_max = float(rear_x.max())
        y_min = float(rear_y.min())
        y_max = float(rear_y.max())
        if x_max > x_min and y_max > y_min:
            cx = rng.uniform(x_min, x_max)
            cy = rng.uniform(y_min, y_max)
        else:
            cx = float(rear_x.mean())
            cy = float(rear_y.mean())
        hx = occ_box_x * 0.5
        hy = occ_box_y * 0.5
        in_box = (
            (x_joint >= cx - hx) & (x_joint <= cx + hx) &
            (y_joint >= cy - hy) & (y_joint <= cy + hy)
        )
        out = pcd[~in_box]
        # occ_box in BEV frame (center in lidar coords before shift)
        occ_box = (cx - joint_shift_x, cy, hx, hy)
        return out, occ_box
    return pcd, None


def apply_trans_aug(pcd, gt_deg, trans_max_m, trans_dir, rng):
    if pcd.shape[0] == 0 or trans_max_m <= 0.0:
        return pcd
    v = np.array([np.cos(np.deg2rad(gt_deg)), np.sin(np.deg2rad(gt_deg))], dtype=np.float32)
    t = rng.uniform(0.0, trans_max_m)
    if trans_dir == "forward":
        sign = 1.0
    elif trans_dir == "backward":
        sign = -1.0
    else:
        sign = 1.0 if rng.random() < 0.5 else -1.0
    out = pcd.copy()
    out[:, 0] += sign * t * v[0]
    out[:, 1] += sign * t * v[1]
    return out


def apply_rot_aug(pcd, rot_deg, joint_shift_x):
    if pcd.shape[0] == 0 or rot_deg == 0.0:
        return pcd
    rot_rad = np.deg2rad(rot_deg)
    cos_r, sin_r = np.cos(rot_rad), np.sin(rot_rad)
    cx = -joint_shift_x
    cy = 0.0
    x = pcd[:, 0] - cx
    y = pcd[:, 1] - cy
    xr = cos_r * x - sin_r * y + cx
    yr = sin_r * x + cos_r * y + cy
    out = pcd.copy()
    out[:, 0] = xr
    out[:, 1] = yr
    return out


def apply_density_downsample(pts_xy, x_range, y_range, res, percentile, rng):
    if pts_xy.shape[0] == 0:
        return pts_xy
    x_min, x_max = x_range
    y_min, y_max = y_range
    H = int(np.ceil((x_max - x_min) / res))
    W = int(np.ceil((y_max - y_min) / res))
    ix = ((pts_xy[:, 0] - x_min) / res).astype(np.int64)
    iy = ((pts_xy[:, 1] - y_min) / res).astype(np.int64)
    ix = np.clip(ix, 0, H - 1)
    iy = np.clip(iy, 0, W - 1)
    counts = np.zeros((H, W), dtype=np.int32)
    np.add.at(counts, (ix, iy), 1)
    cell_counts = counts[ix, iy].astype(np.float32)
    target = np.percentile(cell_counts, percentile)
    target = max(target, 1.0)
    keep_prob = np.minimum(1.0, target / (cell_counts + 1e-6))
    keep = rng.random(pts_xy.shape[0]) < keep_prob
    return pts_xy[keep]


def transform_xy(pts_xy, joint_shift_x, normalize_xy, S, rotate_deg):
    out = pts_xy.copy()
    if normalize_xy:
        out[:, 0] = (out[:, 0] + joint_shift_x) / S
        out[:, 1] = out[:, 1] / S
    else:
        out[:, 0] += joint_shift_x
    if rotate_deg % 360 != 0:
        ang = np.deg2rad(rotate_deg)
        c, s = np.cos(ang), np.sin(ang)
        R = np.array([[c, -s], [s, c]], dtype=np.float32)
        out[:, :2] = (R @ out[:, :2].T).T
    return out


def main():
    args = parse_args()

    if args.config is not None:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
        cfg_dir = os.path.dirname(os.path.abspath(args.config))
        dset_cfg_path = cfg.get("dataset_config")
        if dset_cfg_path:
            if not os.path.isabs(dset_cfg_path):
                cand = os.path.normpath(os.path.join(cfg_dir, dset_cfg_path))
                if os.path.exists(cand):
                    dset_cfg_path = cand
                else:
                    repo_rel = os.path.normpath(os.path.join(os.getcwd(), dset_cfg_path))
                    if os.path.exists(repo_rel):
                        dset_cfg_path = repo_rel
                    else:
                        base = os.path.basename(dset_cfg_path)
                        dset_cfg_path = os.path.normpath(os.path.join(os.getcwd(), "configs", "datasets", base))
            with open(dset_cfg_path, "r") as f:
                dset_cfg = yaml.safe_load(f)
            dset = dset_cfg.get("dataset", {})
            if "root" in dset:
                root_path = dset["root"]
                if not os.path.isabs(root_path):
                    root_path = os.path.normpath(os.path.join(os.getcwd(), root_path))
                args.dataset_root = root_path
            if "split" in dset:
                split_path = dset["split"]
                if not os.path.isabs(split_path):
                    cand = os.path.normpath(os.path.join(cfg_dir, split_path))
                    if os.path.exists(cand):
                        split_path = cand
                    else:
                        split_path = os.path.normpath(os.path.join(os.getcwd(), split_path))
                args.split_dir = os.path.dirname(split_path)

        train_cfg = cfg.get("train", {})
        args.normalize_xy = bool(train_cfg.get("normalize_xy", args.normalize_xy))
        args.dlog_range_norm = bool(train_cfg.get("dlog_range_norm", args.dlog_range_norm))
        args.dlog_range_norm_mode = train_cfg.get("dlog_range_norm_mode", args.dlog_range_norm_mode)
        args.dlog_range_bin_size = float(train_cfg.get("dlog_range_bin_size", args.dlog_range_bin_size))
        args.add_observed_mask = bool(train_cfg.get("add_observed_mask", args.add_observed_mask))
        args.observed_bins = int(train_cfg.get("observed_bins", args.observed_bins))
        args.observed_margin = float(train_cfg.get("observed_margin", args.observed_margin))
        args.occ_binary = bool(train_cfg.get("occ_binary", args.occ_binary))
        args.occ_x_thresh = float(train_cfg.get("occ_x_thresh", args.occ_x_thresh))
        if "occ_y_thresh" in train_cfg:
            args.occ_y_thresh = train_cfg.get("occ_y_thresh")
        args.occ_box_x = float(train_cfg.get("occ_box_x", args.occ_box_x))
        args.occ_box_y = float(train_cfg.get("occ_box_y", args.occ_box_y))
        args.trans_max_m = float(train_cfg.get("trans_max_m", args.trans_max_m))
        args.trans_dir = str(train_cfg.get("trans_dir", args.trans_dir))
        args.rot_deg = float(train_cfg.get("aug_rotate_deg", args.rot_deg))
        exp_name = cfg.get("experiment", {}).get("name", None)
        if args.dlog_stats_path is None and exp_name:
            args.dlog_stats_path = os.path.join(
                "ckpts", exp_name, f"dlog_range_stats_b{args.dlog_range_bin_size}.pt"
            )

    trailer_type = "dummy"
    root = args.dataset_root if "dummy_trailer" in args.dataset_root else os.path.join(args.dataset_root, "dummy_trailer")
    split_json = os.path.join(args.split_dir, "dummy_trailer_split.json")

    frame_dir, gt_deg = select_frame(
        root, split_json, args.split,
        args.angle_deg, args.angle_tol,
        args.max_frames, args.seed
    )
    if frame_dir is None:
        raise RuntimeError("No frame found matching angle criteria.")

    pts = read_pcd(frame_dir)
    if pts.shape[0] == 0:
        raise RuntimeError("Empty point cloud.")

    # (a) reference (aligned view)
    L = TRAILER_TYPES[trailer_type]["len"]
    W = TRAILER_TYPES[trailer_type]["width"]
    S = max(float(L), float(W), 1e-6) / 2.0
    pts_ref = transform_xy(pts.copy(), args.joint_shift_x, args.normalize_xy, S, args.rotate_deg)

    # (b) occlusion aug
    rng = np.random.default_rng(args.occ_seed)
    pts_occ_raw, occ_box = apply_occ_aug(
        pts.copy(), gt_deg, args.joint_shift_x,
        args.occ_x_thresh, args.occ_box_x, args.occ_box_y, rng, args.occ_y_thresh
    )
    pts_occ = transform_xy(pts_occ_raw, args.joint_shift_x, args.normalize_xy, S, args.rotate_deg)

    # (c) translation aug
    rng_t = np.random.default_rng(args.trans_seed)
    pts_trans_raw = apply_trans_aug(pts.copy(), gt_deg, args.trans_max_m, args.trans_dir, rng_t)
    pts_trans = transform_xy(pts_trans_raw, args.joint_shift_x, args.normalize_xy, S, args.rotate_deg)

    # (d) rotation aug
    rng_r = np.random.default_rng(args.rot_seed)
    rot_deg = args.rot_deg
    if rot_deg == 0.0:
        rot_deg = float(rng_r.uniform(-5.0, 5.0))
    pts_rot_raw = apply_rot_aug(pts.copy(), rot_deg, args.joint_shift_x)
    pts_rot = transform_xy(pts_rot_raw, args.joint_shift_x, args.normalize_xy, S, args.rotate_deg)

    xlim = (args.x_min, args.x_max)
    ylim = (args.y_min, args.y_max)
    if args.crop:
        if pts_ref.shape[0] > 0:
            m = (
                (pts_ref[:, 0] >= xlim[0]) & (pts_ref[:, 0] <= xlim[1]) &
                (pts_ref[:, 1] >= ylim[0]) & (pts_ref[:, 1] <= ylim[1])
            )
            pts_ref = pts_ref[m]
        if pts_occ.shape[0] > 0:
            m = (
                (pts_occ[:, 0] >= xlim[0]) & (pts_occ[:, 0] <= xlim[1]) &
                (pts_occ[:, 1] >= ylim[0]) & (pts_occ[:, 1] <= ylim[1])
            )
            pts_occ = pts_occ[m]
        if pts_trans.shape[0] > 0:
            m = (
                (pts_trans[:, 0] >= xlim[0]) & (pts_trans[:, 0] <= xlim[1]) &
                (pts_trans[:, 1] >= ylim[0]) & (pts_trans[:, 1] <= ylim[1])
            )
            pts_trans = pts_trans[m]
        if pts_rot.shape[0] > 0:
            m = (
                (pts_rot[:, 0] >= xlim[0]) & (pts_rot[:, 0] <= xlim[1]) &
                (pts_rot[:, 1] >= ylim[0]) & (pts_rot[:, 1] <= ylim[1])
            )
            pts_rot = pts_rot[m]
    roi_w = float(xlim[1] - xlim[0])
    roi_h = float(ylim[1] - ylim[0])
    panel_h = 3.6
    panel_w = panel_h * (roi_w / roi_h) if roi_h > 0 else 3.6

    fig = plt.figure(figsize=(panel_w * 1.85, panel_h * 2.0), dpi=200, facecolor=args.fig_color)
    gs = fig.add_gridspec(2, 2, hspace=0.16, wspace=0.04)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    axes_all = [ax_a, ax_b, ax_c, ax_d]

    add_panel(ax_a, pts_ref, args.panel_color, xlim, ylim, "(a) Aligned BEV (Reference)",
              args.point_size, args.point_alpha)
    add_panel(ax_b, pts_rot, args.panel_color, xlim, ylim, "(b) Rotation",
              args.point_size, args.point_alpha)
    add_panel(ax_c, pts_trans, args.panel_color, xlim, ylim, "(c) Translation",
              args.point_size, args.point_alpha)
    add_panel(ax_d, pts_occ, args.panel_color, xlim, ylim, "(d) Occlusion",
              args.point_size, args.point_alpha)

    # Visualize occlusion box (dashed)
    # if occ_box is not None:
    #     cx, cy, hx, hy = occ_box
    #     corners = np.array([
    #         [cx - hx, cy - hy],
    #         [cx + hx, cy - hy],
    #         [cx + hx, cy + hy],
    #         [cx - hx, cy + hy],
    #     ], dtype=np.float32)
    #     # rotate occ box by hitch angle around hitch point
    #     theta = np.deg2rad(float(gt_deg))
    #     c, s = np.cos(theta), np.sin(theta)
    #     R = np.array([[c, -s], [s, c]], dtype=np.float32)
    #     hitch = np.array([[-args.joint_shift_x, 0.0]], dtype=np.float32)
    #     corners = (R @ (corners - hitch).T).T + hitch
    #     corners = transform_xy(corners, args.joint_shift_x, args.normalize_xy, S, args.rotate_deg)
    #     occ_poly = plt.Polygon(corners, closed=True, fill=False, edgecolor="#4f7cff",
    #                            linestyle="--", linewidth=1.2)
    #     ax_b.add_patch(occ_poly)

    # Hitch point marker (match figure_aligned_bev: use hitch_x/y in display coords)
    hx, hy = args.hitch_x, args.hitch_y
    if args.rotate_deg % 360 != 0:
        ang = np.deg2rad(args.rotate_deg)
        c, s = np.cos(ang), np.sin(ang)
        R = np.array([[c, -s], [s, c]], dtype=np.float32)
        hx, hy = (R @ np.array([hx, hy], dtype=np.float32)).tolist()
    for ax in axes_all:
        ax.scatter(
            [hx], [hy],
            s=36, facecolors="none", edgecolors="#ff6a00",
            marker="o", linewidths=1.4
        )
        ax.text(
            hx + 0.08, hy + 0.08, "Hitch",
            color="#ff6a00", fontsize=15, ha="left", va="bottom", clip_on=False
        )

    if args.show_ref_ray:
        theta = np.deg2rad(float(args.angle_deg))
        ray_dir = -np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
        ray_dir = ray_dir / (np.linalg.norm(ray_dir) + 1e-8)
        if args.rotate_deg % 360 != 0:
            ang = np.deg2rad(args.rotate_deg)
            c, s = np.cos(ang), np.sin(ang)
            R = np.array([[c, -s], [s, c]], dtype=np.float32)
            ray_dir = (R @ ray_dir.reshape(2, 1)).reshape(2)
        rx = args.hitch_x + ray_dir[0] * args.ray_len
        ry = args.hitch_y + ray_dir[1] * args.ray_len
        for ax in axes_all:
            if ax is ax_b:
                theta_i = np.deg2rad(float(args.angle_deg + rot_deg))
                ray_dir_i = -np.array([np.cos(theta_i), np.sin(theta_i)], dtype=np.float32)
                ray_dir_i = ray_dir_i / (np.linalg.norm(ray_dir_i) + 1e-8)
                if args.rotate_deg % 360 != 0:
                    ang = np.deg2rad(args.rotate_deg)
                    c, s = np.cos(ang), np.sin(ang)
                    R = np.array([[c, -s], [s, c]], dtype=np.float32)
                    ray_dir_i = (R @ ray_dir_i.reshape(2, 1)).reshape(2)
                rx_i = args.hitch_x + ray_dir_i[0] * args.ray_len
                ry_i = args.hitch_y + ray_dir_i[1] * args.ray_len
                ax.annotate(
                    "",
                    xy=(rx_i, ry_i),
                    xytext=(args.hitch_x, args.hitch_y),
                    arrowprops=dict(arrowstyle="->", color="#ffb066", lw=1.2, linestyle="--", alpha=0.6),
                    zorder=6,
                )
            else:
                ax.annotate(
                    "",
                    xy=(rx, ry),
                    xytext=(args.hitch_x, args.hitch_y),
                    arrowprops=dict(arrowstyle="->", color="#ffb066", lw=1.2, linestyle="--", alpha=0.6),
                    zorder=6,
                )

    fig.subplots_adjust(left=0.015, right=0.99, top=0.985, bottom=0.08)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, bbox_inches="tight", facecolor=args.fig_color, transparent=False)
    plt.close(fig)

    print(f"[INFO] Saved: {args.out}")
    print(f"[INFO] Frame: {frame_dir}")
    print(f"[INFO] Hitch angle: {gt_deg:.2f} deg")


if __name__ == "__main__":
    main()
