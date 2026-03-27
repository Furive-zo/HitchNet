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
        description="Aligned BEV figure (before/after) using dataset logic."
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

    # Alignment toggles (after)
    parser.add_argument("--normalize_xy", action="store_true", default=True)
    parser.add_argument("--dlog_range_norm", action="store_true", default=False)
    parser.add_argument("--dlog_range_norm_mode", type=str, default="center")
    parser.add_argument("--dlog_range_bin_size", type=float, default=0.1)
    parser.add_argument("--dlog_stats_path", type=str, default=None)
    parser.add_argument("--add_observed_mask", action="store_true", default=False)
    parser.add_argument("--observed_bins", type=int, default=360)
    parser.add_argument("--observed_margin", type=float, default=0.0)
    parser.add_argument("--occ_binary", action="store_true", default=False)

    # Rendering (match figure_bev_compare)
    parser.add_argument("--panel_color", type=str, default="#e8e8e8")
    parser.add_argument("--fig_color", type=str, default="#ffffff")
    parser.add_argument("--point_size", type=float, default=3.0)
    parser.add_argument("--point_alpha", type=float, default=0.6)
    parser.add_argument("--rotate_deg", type=float, default=90.0,
                        help="Rotate points for display (deg, CCW).")
    parser.add_argument("--crop", action="store_true", default=True,
                        help="Crop points to x/y limits for display.")
    parser.add_argument("--mark_hitch", action="store_true", default=True)
    parser.add_argument("--hitch_x", type=float, default=0.0)
    parser.add_argument("--hitch_y", type=float, default=0.0)
    parser.add_argument("--show_ref_ray", action="store_true", default=True)
    parser.add_argument("--ray_len", type=float, default=1.0)
    parser.add_argument("--out", type=str, default="results/fig_aligned_bev.png")
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


def to_bev(pts, x_range, y_range, res, clip_count, joint_shift_x,
           use_hmax=False, use_dlog=False, add_observed_mask=False,
           observed_bins=360, observed_margin=0.0, occ_binary=False):
    return points_to_bev(
        pts,
        x_range=x_range,
        y_range=y_range,
        z_range=(-2.0, 2.0),
        res=res,
        clip_count=clip_count,
        joint_shift_x=joint_shift_x,
        use_hmax=use_hmax,
        use_dlog=use_dlog,
        occ_binary=occ_binary,
        add_observed_mask=add_observed_mask,
        observed_bins=observed_bins,
        observed_margin=observed_margin,
        add_xy=False,
        add_orient=False,
    )


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


def add_panel(ax, pts_xy, panel_color, xlim, ylim, label, point_size, point_alpha, colors=None):
    ax.set_facecolor(panel_color)
    rect = plt.Rectangle(
        (xlim[0], ylim[0]),
        xlim[1] - xlim[0],
        ylim[1] - ylim[0],
        facecolor=panel_color,
        edgecolor="#d8d8d8",
        linewidth=0.4,
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
    # no label


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


def main():
    args = parse_args()

    # Load settings from config (if provided)
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

    L = TRAILER_TYPES[trailer_type]["len"]
    W = TRAILER_TYPES[trailer_type]["width"]
    S = max(float(L), float(W), 1e-6) / 2.0

    def build_bev(pts_in, do_norm, apply_dlog_norm, apply_vis):
        # Same ROI/res for before/after
        x_range = (args.x_min, args.x_max)
        y_range = (args.y_min, args.y_max)

        if do_norm:
            # dataset logic: shift then scale, and use joint_shift_x=0 in BEV
            pts_use = pts_in.copy()
            pts_use[:, 0] = (pts_use[:, 0] + args.joint_shift_x) / S
            pts_use[:, 1] = pts_use[:, 1] / S
            joint_shift = 0.0
        else:
            pts_use = pts_in
            joint_shift = args.joint_shift_x

        bev = to_bev(
            pts_use, x_range, y_range,
            args.res, args.clip_count, joint_shift_x=joint_shift,
            use_hmax=False, use_dlog=True, add_observed_mask=apply_vis,
            observed_bins=args.observed_bins, observed_margin=args.observed_margin,
            occ_binary=args.occ_binary
        )
        dlog_idx = 1
        dlog = bev[dlog_idx]

        if apply_dlog_norm and args.dlog_stats_path and os.path.exists(args.dlog_stats_path):
            stats = torch.load(args.dlog_stats_path, map_location="cpu", weights_only=False)
            bin_size = float(stats["bin_size"])
            nbins = int(stats["nbins"])
            x_min, x_max = x_range
            y_min, y_max = y_range
            H = int(np.ceil((x_max - x_min) / args.res))
            W2 = int(np.ceil((y_max - y_min) / args.res))
            x_centers = x_min + (np.arange(H) + 0.5) * args.res
            y_centers = y_min + (np.arange(W2) + 0.5) * args.res
            x_grid, y_grid = np.meshgrid(x_centers, y_centers, indexing="ij")
            r = np.sqrt(x_grid ** 2 + y_grid ** 2)
            bin_idx = np.floor(r / bin_size).astype(np.int64)
            bin_idx = np.clip(bin_idx, 0, nbins - 1)
            mu = stats["mu"]
            sigma = stats["sigma"]
            mu_map = mu[bin_idx]
            sig_map = sigma[bin_idx]
            if args.dlog_range_norm_mode == "zscore":
                dlog = (dlog - mu_map) / (sig_map + 1e-6)
            else:
                dlog = dlog - mu_map

        if apply_vis:
            vis_idx = 2
            vis = bev[vis_idx]
            return dlog * vis
        return dlog

    # Before/After scatter views like figure_bev_compare
    pts_before = pts.copy()
    pts_before[:, 0] += args.joint_shift_x

    pts_after = pts.copy()
    L = TRAILER_TYPES[trailer_type]["len"]
    W = TRAILER_TYPES[trailer_type]["width"]
    S = max(float(L), float(W), 1e-6) / 2.0
    pts_after[:, 0] = (pts_after[:, 0] + args.joint_shift_x) / S
    pts_after[:, 1] = pts_after[:, 1] / S

    if args.rotate_deg % 360 != 0:
        ang = np.deg2rad(args.rotate_deg)
        c, s = np.cos(ang), np.sin(ang)
        R = np.array([[c, -s], [s, c]], dtype=np.float32)
        pts_before[:, :2] = (R @ pts_before[:, :2].T).T
        pts_after[:, :2] = (R @ pts_after[:, :2].T).T

    xlim = (args.x_min, args.x_max)
    ylim = (args.y_min, args.y_max)
    if args.crop:
        if pts_before.shape[0] > 0:
            m = (
                (pts_before[:, 0] >= xlim[0]) & (pts_before[:, 0] <= xlim[1]) &
                (pts_before[:, 1] >= ylim[0]) & (pts_before[:, 1] <= ylim[1])
            )
            pts_before = pts_before[m]
        if pts_after.shape[0] > 0:
            m = (
                (pts_after[:, 0] >= xlim[0]) & (pts_after[:, 0] <= xlim[1]) &
                (pts_after[:, 1] >= ylim[0]) & (pts_after[:, 1] <= ylim[1])
            )
            pts_after = pts_after[m]

    roi_w = float(xlim[1] - xlim[0])
    roi_h = float(ylim[1] - ylim[0])
    panel_h = 3.6
    panel_w = panel_h * (roi_w / roi_h) if roi_h > 0 else 3.6
    fig, ax = plt.subplots(1, 1, figsize=(panel_w, panel_h), dpi=200, facecolor=args.fig_color)

    # Density normalization (after only): map dlog to point alpha
    colors_after = None
    if pts_after.shape[0] > 0:
        dlog = dlog_norm_map(
            pts_after,
            x_range=xlim,
            y_range=ylim,
            res=args.res,
            clip_count=args.clip_count,
            occ_binary=args.occ_binary,
            stats_path=args.dlog_stats_path if args.dlog_range_norm else None,
            mode=args.dlog_range_norm_mode,
        )
        dlog_norm = normalize_image(dlog)
        ix = ((pts_after[:, 0] - xlim[0]) / args.res).astype(np.int64)
        iy = ((pts_after[:, 1] - ylim[0]) / args.res).astype(np.int64)
        H, W2 = dlog_norm.shape
        ix = np.clip(ix, 0, H - 1)
        iy = np.clip(iy, 0, W2 - 1)
        a = dlog_norm[ix, iy]

        # visibility-aware mask (hide unobserved)
        vis = None
        if args.add_observed_mask:
            bev_vis = points_to_bev(
                pts_after,
                x_range=xlim,
                y_range=ylim,
                z_range=(-2.0, 2.0),
                res=args.res,
                clip_count=args.clip_count,
                joint_shift_x=0.0,
                use_hmax=False,
                use_dlog=False,
                occ_binary=args.occ_binary,
                add_observed_mask=True,
                observed_bins=args.observed_bins,
                observed_margin=args.observed_margin,
                add_xy=False,
                add_orient=False,
            )
            vis_map = bev_vis[-1]
            vis = vis_map[ix, iy]

        alpha = args.point_alpha * (0.2 + 0.8 * a)
        if vis is not None:
            alpha = alpha * vis

        colors_after = np.zeros((pts_after.shape[0], 4), dtype=np.float32)
        colors_after[:, 3] = alpha

    add_panel(
        ax, pts_after, args.panel_color, xlim, ylim,
        None, args.point_size, args.point_alpha, colors=colors_after
    )

    # Hitch marker + reference ray (match figure_bev_compare style)
    if args.mark_hitch:
        hx, hy = args.hitch_x, args.hitch_y
        if args.rotate_deg % 360 != 0:
            ang = np.deg2rad(args.rotate_deg)
            c, s = np.cos(ang), np.sin(ang)
            R = np.array([[c, -s], [s, c]], dtype=np.float32)
            hx, hy = (R @ np.array([hx, hy], dtype=np.float32)).tolist()
        ax.scatter(
            [hx], [hy],
            s=36, facecolors="none", edgecolors="#ff6a00",
            marker="o", linewidths=1.4
        )
        ax.text(
            hx + 0.08, hy + 0.08, "Hitch",
            color="#ff6a00", fontsize=11, ha="left", va="bottom"
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
        ax.annotate(
            "",
            xy=(rx, ry),
            xytext=(args.hitch_x, args.hitch_y),
            arrowprops=dict(arrowstyle="->", color="#ffb066", lw=1.2, linestyle="--", alpha=0.6),
            zorder=6,
        )

    plt.tight_layout(pad=0.2)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, bbox_inches="tight", facecolor=args.fig_color, transparent=False)
    plt.close(fig)

    print(f"[INFO] Saved: {args.out}")
    print(f"[INFO] Frame: {frame_dir}")
    print(f"[INFO] Hitch angle: {gt_deg:.2f} deg")


if __name__ == "__main__":
    main()
