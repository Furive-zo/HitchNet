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

# Typography for paper figures
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"],
    "font.size": 12,
})


def parse_args():
    parser = argparse.ArgumentParser(
        description="Make a paper-ready BEV point distribution comparison figure."
    )
    parser.add_argument("--dataset_root", type=str, default="datasets/LI-HAE/dataset")
    parser.add_argument("--split_dir", type=str, default="datasets/LI-HAE/splits")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--angle_deg", type=float, required=True, help="Target hitch angle (deg).")
    parser.add_argument("--angle_tol", type=float, default=2.0, help="Tolerance (deg).")
    parser.add_argument("--max_frames", type=int, default=300, help="Max frames to scan per trailer.")
    parser.add_argument("--single_frame", action="store_true", help="Use a single matching frame per trailer.")
    parser.add_argument("--max_points", type=int, default=200000, help="Max points to plot per trailer.")
    parser.add_argument("--joint_shift_x", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--x_min", type=float, default=-2.5)
    parser.add_argument("--x_max", type=float, default=2.5)
    parser.add_argument("--y_min", type=float, default=-4.0)
    parser.add_argument("--y_max", type=float, default=0.5)
    parser.add_argument("--point_size", type=float, default=3.0)
    parser.add_argument("--point_alpha", type=float, default=0.6)
    parser.add_argument("--rotate_deg", type=float, default=90.0, help="Rotate points (deg) counter-clockwise.")
    parser.add_argument("--fig_w", type=float, default=11.5)
    parser.add_argument("--fig_h", type=float, default=3.8)
    parser.add_argument("--crop", action="store_true", help="Crop points to x/y limits before plotting.")
    parser.add_argument("--panel_color", type=str, default="#e8e8e8")
    parser.add_argument("--panel_border_color", type=str, default="#2f2f2f")
    parser.add_argument("--panel_border_width", type=float, default=0.9)
    parser.add_argument("--fig_color", type=str, default="#ffffff")
    parser.add_argument("--mark_hitch", action="store_true", help="Draw hitch point marker.")
    parser.add_argument("--hitch_x", type=float, default=-0.0)
    parser.add_argument("--hitch_y", type=float, default=0.0)
    parser.add_argument("--show_ref_ray", action="store_true", default=True,
                        help="Draw reference ray for hitch angle.")
    parser.add_argument("--ray_len", type=float, default=1.0)
    parser.add_argument("--stats_res", type=float, default=0.1, help="Grid resolution for occupied-ratio/entropy stats.")
    parser.add_argument("--stats_fontsize", type=float, default=16.0)
    parser.add_argument("--out", type=str, default="results/fig_bev_compare.png")
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


def read_pcd_xy(frame_dir, joint_shift_x):
    pcd_path = os.path.join(frame_dir, "trailer_point.pcd")
    pcd = o3d.io.read_point_cloud(pcd_path)
    pts = np.asarray(pcd.points, dtype=np.float32)
    if pts.shape[0] == 0:
        return pts.reshape(0, 2)
    pts = pts.copy()
    pts[:, 0] += joint_shift_x
    return pts[:, :2]


def collect_points(root, split_json, split, target_deg, tol_deg,
                   max_frames, max_points, joint_shift_x, seed, single_frame=False):
    seqs = load_split_sequences(split_json, split)
    frames = list_frames(root, seqs)
    rng = np.random.default_rng(seed)
    rng.shuffle(frames)

    kept = []
    used_frames = 0
    for fr in frames:
        if max_frames > 0 and used_frames >= max_frames:
            break
        angle = read_angle_deg(fr)
        if abs(angle - target_deg) > tol_deg:
            continue
        xy = read_pcd_xy(fr, joint_shift_x)
        if xy.shape[0] == 0:
            continue
        kept.append(xy)
        used_frames += 1
        if single_frame and len(kept) >= 1:
            break
        if max_points > 0 and sum(k.shape[0] for k in kept) >= max_points:
            break

    if len(kept) == 0:
        return np.zeros((0, 2), dtype=np.float32), used_frames

    pts = np.concatenate(kept, axis=0)
    if max_points > 0 and pts.shape[0] > max_points:
        idx = rng.choice(pts.shape[0], size=max_points, replace=False)
        pts = pts[idx]
    return pts, used_frames


def plot_panel(
    ax, pts, title_below, xlim, ylim, point_size, point_alpha,
    hitch_xy=None, panel_color="#7fb8ff", hitch_label=False, ref_ray=None,
    border_color="#2f2f2f", border_width=0.9, stats_text=None, stats_fontsize=10.0
):
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    # Explicit background rectangle to avoid backend/alpha issues
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
    if pts.shape[0] > 0:
        ax.scatter(
            pts[:, 0], pts[:, 1],
            s=point_size, c="black", alpha=point_alpha, linewidths=0
        )
    if hitch_xy is not None:
        ax.scatter(
            [hitch_xy[0]], [hitch_xy[1]],
            s=28, facecolors="none", edgecolors="#ff6a00",
            marker="o", linewidths=1.4
        )
        if hitch_label:
            ax.text(
                hitch_xy[0] + 0.08, hitch_xy[1] + 0.08, "Hitch",
                color="#ff6a00", fontsize=14, ha="left", va="bottom"
            )
    if ref_ray is not None and hitch_xy is not None:
        ax.annotate(
            "",
            xy=(ref_ray[0], ref_ray[1]),
            xytext=(hitch_xy[0], hitch_xy[1]),
            arrowprops=dict(arrowstyle="->", color="#ffb066", lw=1.2, linestyle="--", alpha=0.9),
            zorder=6,
        )
    ax.set_aspect("equal", "box")
    ax.axis("off")
    if title_below:
        ax.text(0.5, -0.04, title_below, transform=ax.transAxes,
                ha="center", va="top", fontsize=20)
    if stats_text:
        ax.text(
            0.02, 0.02, stats_text, transform=ax.transAxes,
            ha="left", va="bottom", fontsize=stats_fontsize, color="#202020",
            bbox=dict(facecolor="white", alpha=0.1, edgecolor="none", pad=1.8),
            zorder=25,
        )
    # Panel border in axes coordinates (visible even when axis is off).
    ax.add_patch(
        plt.Rectangle(
            (0, 0), 1, 1,
            transform=ax.transAxes,
            fill=False,
            edgecolor=border_color,
            linewidth=border_width,
            zorder=20,
            clip_on=False,
        )
    )


def compute_panel_stats(pts, xlim, ylim, res):
    n_pts = int(pts.shape[0])
    if n_pts == 0:
        return 0, 0.0, 0.0

    x_edges = np.arange(xlim[0], xlim[1] + res, res, dtype=np.float32)
    y_edges = np.arange(ylim[0], ylim[1] + res, res, dtype=np.float32)
    if x_edges.size < 2 or y_edges.size < 2:
        return n_pts, 0.0, 0.0

    hist, _, _ = np.histogram2d(pts[:, 0], pts[:, 1], bins=[x_edges, y_edges])
    occ = hist > 0
    occ_ratio = float(occ.sum()) / float(hist.size)

    occ_counts = hist[occ]
    p = occ_counts / (occ_counts.sum() + 1e-12)
    entropy = float(-(p * np.log2(p + 1e-12)).sum())
    return n_pts, occ_ratio, entropy


def format_panel_stats(pts, xlim, ylim, res):
    n_pts, occ_ratio, entropy = compute_panel_stats(pts, xlim, ylim, res)
    return f"Points: {n_pts}\nOccupied: {occ_ratio*100:.1f}%\nEntropy: {entropy:.2f}"


def main():
    args = parse_args()

    charger_root = os.path.join(args.dataset_root, "charger_trailer")
    dummy_root = os.path.join(args.dataset_root, "dummy_trailer")
    temp_root = os.path.join(args.dataset_root, "temporary_trailer")
    charger_split = os.path.join(args.split_dir, "charger_trailer_split.json")
    dummy_split = os.path.join(args.split_dir, "dummy_trailer_split.json")
    temp_split = os.path.join(args.split_dir, "temporary_trailer_split.json")

    pts_charger, frames_charger = collect_points(
        charger_root, charger_split, args.split,
        args.angle_deg, args.angle_tol,
        args.max_frames, args.max_points, args.joint_shift_x, args.seed, args.single_frame
    )
    pts_temp, frames_temp = collect_points(
        temp_root, temp_split, args.split,
        args.angle_deg, args.angle_tol,
        args.max_frames, args.max_points, args.joint_shift_x, args.seed + 1, args.single_frame
    )
    pts_dummy, frames_dummy = collect_points(
        dummy_root, dummy_split, args.split,
        args.angle_deg, args.angle_tol,
        args.max_frames, args.max_points, args.joint_shift_x, args.seed + 2, args.single_frame
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(args.fig_w, args.fig_h), dpi=200, facecolor=args.fig_color)
    fig.patch.set_facecolor(args.fig_color)
    fig.patch.set_alpha(1.0)
    xlim = (args.x_min, args.x_max)
    ylim = (args.y_min, args.y_max)

    if args.rotate_deg % 360 != 0:
        ang = np.deg2rad(args.rotate_deg)
        c, s = np.cos(ang), np.sin(ang)
        R = np.array([[c, -s], [s, c]], dtype=np.float32)
        if pts_charger.shape[0] > 0:
            pts_charger = (R @ pts_charger.T).T
        if pts_temp.shape[0] > 0:
            pts_temp = (R @ pts_temp.T).T
        if pts_dummy.shape[0] > 0:
            pts_dummy = (R @ pts_dummy.T).T

    if args.crop:
        if pts_charger.shape[0] > 0:
            m = (
                (pts_charger[:, 0] >= xlim[0]) & (pts_charger[:, 0] <= xlim[1]) &
                (pts_charger[:, 1] >= ylim[0]) & (pts_charger[:, 1] <= ylim[1])
            )
            pts_charger = pts_charger[m]
        if pts_temp.shape[0] > 0:
            m = (
                (pts_temp[:, 0] >= xlim[0]) & (pts_temp[:, 0] <= xlim[1]) &
                (pts_temp[:, 1] >= ylim[0]) & (pts_temp[:, 1] <= ylim[1])
            )
            pts_temp = pts_temp[m]
        if pts_dummy.shape[0] > 0:
            m = (
                (pts_dummy[:, 0] >= xlim[0]) & (pts_dummy[:, 0] <= xlim[1]) &
                (pts_dummy[:, 1] >= ylim[0]) & (pts_dummy[:, 1] <= ylim[1])
            )
            pts_dummy = pts_dummy[m]

    hitch_xy = None
    if args.mark_hitch:
        hitch_xy = np.array([args.hitch_x, args.hitch_y], dtype=np.float32)
        if args.rotate_deg % 360 != 0:
            hitch_xy = (R @ hitch_xy.reshape(2, 1)).reshape(2)

    ref_ray = None
    if args.show_ref_ray and hitch_xy is not None:
        theta = np.deg2rad(float(args.angle_deg))
        ray_dir = -np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
        if args.rotate_deg % 360 != 0:
            ray_dir = (R @ ray_dir.reshape(2, 1)).reshape(2)
        ray_dir = ray_dir / (np.linalg.norm(ray_dir) + 1e-8)

        # Clamp ray to panel bounds so it's always visible
        x0, y0 = float(hitch_xy[0]), float(hitch_xy[1])
        dx, dy = float(ray_dir[0]), float(ray_dir[1])
        t_candidates = []
        if abs(dx) > 1e-6:
            t_candidates.append((xlim[0] - x0) / dx)
            t_candidates.append((xlim[1] - x0) / dx)
        if abs(dy) > 1e-6:
            t_candidates.append((ylim[0] - y0) / dy)
            t_candidates.append((ylim[1] - y0) / dy)
        t_pos = [t for t in t_candidates if t > 0]
        t_max = min(t_pos) if t_pos else float(args.ray_len)
        t = min(float(args.ray_len), t_max)
        if t <= 0.0:
            t = min(float(args.ray_len), 0.5)
        ref_ray = hitch_xy + ray_dir * t

    plot_panel(
        axes[0], pts_charger, "(a) Short-Tall trailer",
        xlim, ylim, args.point_size, args.point_alpha,
        hitch_xy=hitch_xy, panel_color=args.panel_color,
        hitch_label=True, ref_ray=ref_ray,
        border_color=args.panel_border_color, border_width=args.panel_border_width,
        stats_text=format_panel_stats(pts_charger, xlim, ylim, args.stats_res),
        stats_fontsize=args.stats_fontsize,
    )
    plot_panel(
        axes[1], pts_temp, "(b) Compact trailer",
        xlim, ylim, args.point_size, args.point_alpha,
        hitch_xy=hitch_xy, panel_color=args.panel_color,
        hitch_label=True, ref_ray=ref_ray,
        border_color=args.panel_border_color, border_width=args.panel_border_width,
        stats_text=format_panel_stats(pts_temp, xlim, ylim, args.stats_res),
        stats_fontsize=args.stats_fontsize,
    )
    plot_panel(
        axes[2], pts_dummy, "(c) Long-Flat trailer",
        xlim, ylim, args.point_size, args.point_alpha,
        hitch_xy=hitch_xy, panel_color=args.panel_color,
        hitch_label=True, ref_ray=ref_ray,
        border_color=args.panel_border_color, border_width=args.panel_border_width,
        stats_text=format_panel_stats(pts_dummy, xlim, ylim, args.stats_res),
        stats_fontsize=args.stats_fontsize,
    )

    plt.tight_layout(pad=0.2, w_pad=0.6)
    plt.savefig(args.out, bbox_inches="tight", facecolor=args.fig_color, transparent=False)
    plt.close(fig)

    print(f"[INFO] Saved: {args.out}")
    print(f"[INFO] charger_trailer frames used: {frames_charger}, points: {pts_charger.shape[0]}")
    print(f"[INFO] dummy_trailer frames used: {frames_dummy}, points: {pts_dummy.shape[0]}")
    print(f"[INFO] temporary_trailer frames used: {frames_temp}, points: {pts_temp.shape[0]}")


if __name__ == "__main__":
    main()
