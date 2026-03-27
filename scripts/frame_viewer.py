#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import json
import argparse
from pathlib import Path

import numpy as np
import open3d as o3d

import matplotlib


def parse_args():
    p = argparse.ArgumentParser(description="Frame BEV viewer for quick sample selection.")
    p.add_argument("--dataset_root", type=str, default="datasets/LI-HAE/dataset")
    p.add_argument("--split_dir", type=str, default="datasets/LI-HAE/splits")
    p.add_argument("--trailer_type", type=str, choices=["charger", "dummy", "temporary"], default="charger")
    p.add_argument("--split", type=str, choices=["train", "val", "test"], default="test")
    p.add_argument("--seq_contains", type=str, default="", help="Filter sequence name by substring.")
    p.add_argument("--target_angle_deg", type=float, default=None, help="Sort by closeness to this angle.")
    p.add_argument("--angle_tol_deg", type=float, default=None, help="Optional angle filter around target.")
    p.add_argument("--max_frames", type=int, default=60)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ncols", type=int, default=5)
    p.add_argument("--point_size", type=float, default=1.8)
    p.add_argument("--point_alpha", type=float, default=0.65)
    p.add_argument("--panel_color", type=str, default="#e8e8e8")
    p.add_argument("--fig_color", type=str, default="#ffffff")
    p.add_argument("--joint_shift_x", type=float, default=0.8)
    p.add_argument("--rotate_deg", type=float, default=90.0)
    p.add_argument("--x_min", type=float, default=-2.5)
    p.add_argument("--x_max", type=float, default=2.5)
    p.add_argument("--y_min", type=float, default=-4.0)
    p.add_argument("--y_max", type=float, default=0.5)
    p.add_argument("--out_png", type=str, default="results/frame_viewer.png")
    p.add_argument("--out_csv", type=str, default="results/frame_viewer_list.csv")
    p.add_argument("--pick_frame", type=str, default="",
                   help="Interactive pick mode: seq/frame_name (e.g., scurve_5/frame_000563).")
    p.add_argument("--pick_out", type=str, default="results/frame_pick_points.csv",
                   help="Output CSV path for picked points/boxes.")
    return p.parse_args()


def load_split_sequences(split_json, split):
    with open(split_json, "r") as f:
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


def read_pcd_xy(frame_dir):
    pcd_path = os.path.join(frame_dir, "trailer_point.pcd")
    pcd = o3d.io.read_point_cloud(pcd_path)
    pts = np.asarray(pcd.points, dtype=np.float32)
    if pts.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return pts[:, :2]


def transform_xy(pts_xy, joint_shift_x, rotate_deg):
    out = pts_xy.copy()
    out[:, 0] += float(joint_shift_x)
    if rotate_deg % 360 != 0 and out.shape[0] > 0:
        ang = np.deg2rad(float(rotate_deg))
        c, s = np.cos(ang), np.sin(ang)
        R = np.array([[c, -s], [s, c]], dtype=np.float32)
        out = (R @ out.T).T
    return out


def parse_seq_frame(spec):
    s = (spec or "").strip()
    if not s:
        return None, None
    if "/" not in s:
        raise ValueError(f"Invalid --pick_frame: {s}. Use seq/frame_name")
    seq, frame = s.split("/", 1)
    return seq.strip(), frame.strip()


def pick_points_interactive(plt, pts, xlim, ylim, panel_color, fig_color, point_size, point_alpha, out_csv):
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 6.2), dpi=160)
    fig.patch.set_facecolor(fig_color)
    ax.set_facecolor(panel_color)
    if pts.shape[0] > 0:
        ax.scatter(pts[:, 0], pts[:, 1], s=point_size, c="black", alpha=point_alpha, linewidths=0)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal", "box")
    ax.grid(alpha=0.2, linestyle="--")
    ax.set_title("Click points (x,y in meters). Press Enter when done.")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    clicked = plt.ginput(n=-1, timeout=0, show_clicks=True)
    points = np.asarray(clicked, dtype=np.float64) if len(clicked) else np.zeros((0, 2), dtype=np.float64)
    if points.shape[0] > 0:
        for i, (x, y) in enumerate(points):
            ax.text(x, y, str(i), color="#d62828", fontsize=9, ha="left", va="bottom")
        plt.draw()

    plt.show(block=False)
    plt.pause(0.4)
    plt.close(fig)

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["idx", "x_m", "y_m"])
        w.writeheader()
        for i, (x, y) in enumerate(points):
            w.writerow({"idx": i, "x_m": float(x), "y_m": float(y)})

    print(f"[INFO] Saved picked points: {out_csv}")
    if points.shape[0] >= 2:
        print("[INFO] Suggested erase boxes (pair points as corners):")
        for i in range(0, points.shape[0] - 1, 2):
            x0, y0 = points[i]
            x1, y1 = points[i + 1]
            if x1 < x0:
                x0, x1 = x1, x0
            if y1 < y0:
                y0, y1 = y1, y0
            print(f"  {x0:.3f},{x1:.3f},{y0:.3f},{y1:.3f}")


def main():
    args = parse_args()
    # Backend: interactive for pick mode, file-only for grid mode.
    if args.pick_frame:
        has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        if has_display:
            try:
                matplotlib.use("TkAgg")
            except Exception:
                try:
                    matplotlib.use("Qt5Agg")
                except Exception:
                    matplotlib.use("Agg")
        else:
            matplotlib.use("Agg")
    else:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(args.seed)

    root = os.path.join(args.dataset_root, f"{args.trailer_type}_trailer")
    split_json = os.path.join(args.split_dir, f"{args.trailer_type}_trailer_split.json")
    seqs = load_split_sequences(split_json, args.split)
    if args.seq_contains:
        key = args.seq_contains.lower()
        seqs = [s for s in seqs if key in s.lower()]
    frames = list_frames(root, seqs)
    if len(frames) == 0:
        raise RuntimeError("No frames found after filtering.")

    rows = []
    for fr in frames:
        try:
            ang = read_angle_deg(fr)
        except Exception:
            continue
        if args.target_angle_deg is not None and args.angle_tol_deg is not None:
            if abs(ang - float(args.target_angle_deg)) > float(args.angle_tol_deg):
                continue
        rows.append({"frame_dir": fr, "angle_deg": ang})

    if len(rows) == 0:
        raise RuntimeError("No frames left after angle filtering.")

    if args.target_angle_deg is not None:
        rows.sort(key=lambda r: abs(r["angle_deg"] - float(args.target_angle_deg)))
    else:
        rng.shuffle(rows)

    rows = rows[::max(int(args.stride), 1)]
    rows = rows[: max(int(args.max_frames), 1)]

    # Interactive single-frame pick mode
    if args.pick_frame:
        seq_pick, frame_pick = parse_seq_frame(args.pick_frame)
        hit = None
        for r in rows:
            fr = r["frame_dir"]
            if Path(fr).parent.name == seq_pick and Path(fr).name == frame_pick:
                hit = r
                break
        if hit is None:
            # fallback search from full frame list
            for fr in frames:
                if Path(fr).parent.name == seq_pick and Path(fr).name == frame_pick:
                    hit = {"frame_dir": fr, "angle_deg": read_angle_deg(fr)}
                    break
        if hit is None:
            raise RuntimeError(f"--pick_frame not found: {args.pick_frame}")

        pts = read_pcd_xy(hit["frame_dir"])
        pts = transform_xy(pts, args.joint_shift_x, args.rotate_deg)
        if pts.shape[0] > 0:
            m = (
                (pts[:, 0] >= args.x_min) & (pts[:, 0] <= args.x_max) &
                (pts[:, 1] >= args.y_min) & (pts[:, 1] <= args.y_max)
            )
            pts = pts[m]
        print(f"[INFO] Pick mode frame: {args.pick_frame} | GT={hit['angle_deg']:.2f} deg")
        if matplotlib.get_backend().lower() == "agg":
            # Headless fallback: save static image and ask user to run with GUI backend/session.
            fig, ax = plt.subplots(1, 1, figsize=(7.2, 6.2), dpi=160)
            fig.patch.set_facecolor(args.fig_color)
            ax.set_facecolor(args.panel_color)
            if pts.shape[0] > 0:
                ax.scatter(pts[:, 0], pts[:, 1], s=args.point_size, c="black", alpha=args.point_alpha, linewidths=0)
            ax.set_xlim((args.x_min, args.x_max))
            ax.set_ylim((args.y_min, args.y_max))
            ax.set_aspect("equal", "box")
            ax.grid(alpha=0.2, linestyle="--")
            ax.set_title(f"Headless mode: {args.pick_frame}")
            os.makedirs(os.path.dirname(args.out_png), exist_ok=True)
            fig.savefig(args.out_png, facecolor=args.fig_color)
            plt.close(fig)
            print(f"[WARN] Interactive click requires GUI backend. Saved static view: {args.out_png}")
            return
        pick_points_interactive(
            plt=plt,
            pts=pts,
            xlim=(args.x_min, args.x_max),
            ylim=(args.y_min, args.y_max),
            panel_color=args.panel_color,
            fig_color=args.fig_color,
            point_size=args.point_size,
            point_alpha=args.point_alpha,
            out_csv=args.pick_out,
        )
        return

    n = len(rows)
    ncols = max(1, int(args.ncols))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.1 * nrows), dpi=180)
    fig.patch.set_facecolor(args.fig_color)
    axes = np.array(axes).reshape(nrows, ncols)

    xlim = (args.x_min, args.x_max)
    ylim = (args.y_min, args.y_max)

    for i in range(nrows * ncols):
        ax = axes.flat[i]
        ax.set_facecolor(args.panel_color)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_linewidth(0.6)
            sp.set_edgecolor("#444444")
        if i >= n:
            ax.axis("off")
            continue

        fr = rows[i]["frame_dir"]
        ang = rows[i]["angle_deg"]
        seq = Path(fr).parent.name
        frm = Path(fr).name
        pts = read_pcd_xy(fr)
        pts = transform_xy(pts, args.joint_shift_x, args.rotate_deg)

        if pts.shape[0] > 0:
            m = (
                (pts[:, 0] >= xlim[0]) & (pts[:, 0] <= xlim[1]) &
                (pts[:, 1] >= ylim[0]) & (pts[:, 1] <= ylim[1])
            )
            pts = pts[m]
            if pts.shape[0] > 0:
                ax.scatter(pts[:, 0], pts[:, 1], s=args.point_size, c="black", alpha=args.point_alpha, linewidths=0)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal", "box")
        ax.set_title(f"[{i}] {seq}/{frm}\nGT {ang:.1f}°", fontsize=9, pad=2)

        rows[i]["view_index"] = i
        rows[i]["sequence"] = seq
        rows[i]["frame"] = frm

    fig.subplots_adjust(left=0.01, right=0.995, top=0.99, bottom=0.01, hspace=0.20, wspace=0.06)
    os.makedirs(os.path.dirname(args.out_png), exist_ok=True)
    fig.savefig(args.out_png, facecolor=args.fig_color)
    plt.close(fig)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["view_index", "sequence", "frame", "angle_deg", "frame_dir"])
        w.writeheader()
        w.writerows(rows)

    print(f"[INFO] Saved viewer image: {args.out_png}")
    print(f"[INFO] Saved frame list : {args.out_csv}")
    print(f"[INFO] Frames shown: {n}")


if __name__ == "__main__":
    main()
