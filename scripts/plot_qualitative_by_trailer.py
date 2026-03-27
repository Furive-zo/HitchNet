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
from matplotlib.patches import Rectangle
from matplotlib.path import Path as MplPath

from utils.load_config import load_config
from utils.load_dataset import HitchDataset, TRAILER_TYPES
from utils.collate import collate_fn
from utils.angle import wrap_rad_torch
from models import build_model


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"],
    "font.size": 12,
})

TRAILER_ORDER = ["charger", "temporary", "dummy"]
TRAILER_COL_TITLE = {
    "charger": "Short--Tall",
    "temporary": "Compact",
    "dummy": "Long--Flat",
}
COL_SUBTITLE = {
    "charger": "(Self-occlusion)",
    "temporary": "(Sparse LiDAR returns)",
    "dummy": "(Out-of-training angle range)",
}

METHOD_ORDER = ["Naive BEV", "CORAL-DG", "Ours(Alignment only)", "Ours(Full)"]
MODEL_CFG_BY_TRAILER = {
    "charger": {
        "Naive BEV": "configs/experiments/dummy_bev_resnet_regression.yaml",
        "CORAL-DG": "configs/experiments/dummy_temporary_bev_resnet_regression_coral_dg.yaml",
        "Ours(Alignment only)": "configs/experiments/dummy_bev_resnet_regression_norm.yaml",
        "Ours(Full)": "configs/experiments/dummy_bev_resnet_regression_norm_aug.yaml",
    },
    "dummy": {
        "Naive BEV": "configs/experiments/charger_bev_resnet_regression.yaml",
        "CORAL-DG": "configs/experiments/charger_temporary_bev_resnet_regression_coral_dg.yaml",
        "Ours(Alignment only)": "configs/experiments/charger_bev_resnet_regression_norm.yaml",
        "Ours(Full)": "configs/experiments/charger_bev_resnet_regression_norm_aug.yaml",
    },
    "temporary": {
        "Naive BEV": "configs/experiments/charger_bev_resnet_regression.yaml",
        "CORAL-DG": "configs/experiments/charger_dummy_bev_resnet_regression_coral_dg.yaml",
        "Ours(Alignment only)": "configs/experiments/charger_bev_resnet_regression_norm.yaml",
        "Ours(Full)": "configs/experiments/charger_bev_resnet_regression_norm_aug.yaml",
    },
}


def parse_args():
    p = argparse.ArgumentParser(description="Qualitative 4x3 figure (rows=methods, cols=trailers)")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--top_percent", type=float, default=5.0)
    p.add_argument("--top_percent_short_compact", type=float, default=10.0)
    p.add_argument("--min_abs_gt_deg", type=float, default=25.0)
    p.add_argument("--viz_joint_shift_x", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fig_w", type=float, default=11.2)
    p.add_argument("--fig_h", type=float, default=9.5)
    p.add_argument("--panel_color", type=str, default="#e8e8e8")
    p.add_argument("--fig_color", type=str, default="#ffffff")
    p.add_argument("--border_color", type=str, default="#3f3f3f")
    p.add_argument("--border_width", type=float, default=1.0)
    p.add_argument("--compact_box_len_scale", type=float, default=0.82,
                   help="Extra length scale for Compact(temporary) trailer box.")
    p.add_argument("--compact_box_wid_scale", type=float, default=0.92,
                   help="Extra width scale for Compact(temporary) trailer box.")
    p.add_argument("--short_tall_box_len_scale", type=float, default=0.94,
                   help="Extra length scale for Short--Tall(charger) trailer box.")
    p.add_argument("--short_tall_box_wid_scale", type=float, default=0.94,
                   help="Extra width scale for Short--Tall(charger) trailer box.")
    p.add_argument("--fixed_frame_charger", type=str, default="scurve_5/frame_000563",
                   help="Fixed frame for Short--Tall column: seq/frame_name (e.g., scurve_5/frame_001234).")
    p.add_argument("--fixed_frame_temporary", type=str, default="reverse_5/frame_000983",
                   help="Fixed frame for Compact column: seq/frame_name (e.g., urban_5/frame_000987).")
    p.add_argument("--fixed_frame_dummy", type=str, default="urban_7/frame_001504",
                   help="Fixed frame for Long--Flat column: seq/frame_name.")
    p.add_argument("--occlude_short_tall", action="store_true", default=True,
                   help="Apply a synthetic occlusion patch to Short--Tall column for visualization.")
    p.add_argument("--occ_cx_frac", type=float, default=0.40, help="Occlusion patch center x (0..1).")
    p.add_argument("--occ_cy_frac", type=float, default=0.33, help="Occlusion patch center y (0..1).")
    p.add_argument("--occ_w_frac", type=float, default=0.22, help="Occlusion patch width (0..1).")
    p.add_argument("--occ_h_frac", type=float, default=0.18, help="Occlusion patch height (0..1).")
    p.add_argument("--out", type=str, default="results/qualitative_compare_4x3.png")
    return p.parse_args()


def move_batch_to_device(batch, device):
    return {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


def box_corners(theta, length, width):
    corners = np.array([[-length, -width / 2], [0.0, -width / 2], [0.0, width / 2], [-length, width / 2]], dtype=np.float32)
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    return corners @ rot.T


def corners_to_pixels(corners, x_min, y_min, res):
    ix = (corners[:, 0] - x_min) / res
    iy = (corners[:, 1] - y_min) / res
    return np.stack([iy, ix], axis=1)


def shift_bev_x(bev, shift_m, res):
    px = int(np.round(float(shift_m) / float(res)))
    if px == 0:
        return bev
    out = np.zeros_like(bev)
    if px > 0:
        out[px:, :] = bev[:-px, :]
    else:
        k = -px
        out[:-k, :] = bev[k:, :]
    return out


def apply_occ_patch(bev, cx_frac, cy_frac, w_frac, h_frac):
    h, w = bev.shape
    cx = int(np.clip(cx_frac, 0.0, 1.0) * (h - 1))
    cy = int(np.clip(cy_frac, 0.0, 1.0) * (w - 1))
    hw = max(1, int(0.5 * np.clip(w_frac, 0.0, 1.0) * h))
    hh = max(1, int(0.5 * np.clip(h_frac, 0.0, 1.0) * w))
    x0 = max(0, cx - hw)
    x1 = min(h, cx + hw)
    y0 = max(0, cy - hh)
    y1 = min(w, cy + hh)
    out = bev.copy()
    out[x0:x1, y0:y1] = 0.0
    return out


def polygon_inside_mask(shape_hw, poly_xy):
    h, w = shape_hw
    yy, xx = np.mgrid[0:h, 0:w]
    pts = np.stack([xx.ravel(), yy.ravel()], axis=1)
    return MplPath(poly_xy).contains_points(pts).reshape(h, w)


def build_dataset_for_model(cfg_path, eval_trailer_type, batch_size_override, num_workers_override):
    cfg = load_config(cfg_path)
    cfg_dir = os.path.dirname(cfg_path)

    ds_rel = f"datasets/{eval_trailer_type}.yaml"
    ds_path = os.path.normpath(os.path.join(cfg_dir, "..", ds_rel))
    with open(ds_path, "r") as f:
        ds_cfg = yaml.safe_load(f)
    cfg["dataset"] = ds_cfg.get("dataset", {})

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

    exp_name = cfg.get("experiment", {}).get("name", Path(cfg_path).stem)
    if bool(train_cfg.get("dlog_range_norm", False)):
        bin_size = float(train_cfg.get("dlog_range_bin_size", 0.1))
        stats_path = train_cfg.get("dlog_stats_path", os.path.join("ckpts", exp_name, f"dlog_range_stats_b{bin_size}.pt"))
        if os.path.exists(stats_path):
            stats = torch.load(stats_path, map_location="cpu", weights_only=False)
            dataset.dlog_range_stats = stats
            dataset._build_range_bin_idx()

    batch_size = batch_size_override if batch_size_override is not None else train_cfg.get("batch_size", 8)
    num_workers = num_workers_override if num_workers_override is not None else dset_cfg.get("num_workers", 4)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)
    return cfg, dataset, loader, exp_name


def evaluate_model(cfg_path, ckpt_override, eval_trailer_type, device, batch_size, num_workers):
    cfg, dataset, loader, exp_name = build_dataset_for_model(cfg_path, eval_trailer_type, batch_size, num_workers)
    model_cfg = cfg["model"]
    ckpt_path = ckpt_override or os.path.join("ckpts", exp_name, "best.pth")

    model = build_model(model_cfg).to(device)
    if model_cfg.get("name") not in ("rule_based", "rule_based_pca", "rule_based_ols", "rule_based_mle"):
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt, strict=True)
    model.eval()

    pred_deg, gt_deg, err_deg, frame_names = [], [], [], []
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

    return {
        "cfg": cfg,
        "dataset": dataset,
        "ckpt": ckpt_path,
        "pred_deg": np.array(pred_deg, dtype=np.float32),
        "gt_deg": np.array(gt_deg, dtype=np.float32),
        "err_deg": np.array(err_deg, dtype=np.float32),
        "frame_names": frame_names,
    }


def pick_index(proposed_err_abs, gt_deg, top_percent, min_abs_gt_deg, seed):
    n = len(proposed_err_abs)
    top_k = max(1, int(np.ceil(n * top_percent / 100.0)))
    order = np.argsort(-proposed_err_abs)
    pool = order[:top_k]
    if min_abs_gt_deg > 0:
        large_angle = [i for i in pool.tolist() if abs(float(gt_deg[i])) >= float(min_abs_gt_deg)]
        if len(large_angle) > 0:
            pool = np.asarray(large_angle, dtype=np.int64)
        else:
            # Fallback: keep high-error pool but prioritize larger |GT| entries.
            pool = np.asarray(
                sorted(pool.tolist(), key=lambda i: abs(float(gt_deg[i])), reverse=True)[:max(1, min(20, len(pool)))],
                dtype=np.int64,
            )
    rng = np.random.default_rng(seed)
    return int(rng.choice(pool))


def find_frame_index(frame_names, target_seq, target_frame):
    norm_seq = target_seq.replace("_", "").lower()
    for i, fr in enumerate(frame_names):
        seq = Path(fr).parent.name
        frm = Path(fr).name
        if frm == target_frame and seq.replace("_", "").lower() == norm_seq:
            return i
    return None


def parse_seq_frame(spec):
    s = (spec or "").strip()
    if not s:
        return None, None
    if "/" not in s:
        raise ValueError(f"Invalid fixed frame spec: {s}. Use seq/frame_name.")
    seq, frame = s.split("/", 1)
    return seq.strip(), frame.strip()


def plot_cell(
    ax, bev, gt_deg, pr_deg, err_deg, dataset, trailer_type, normalize_xy, viz_joint_shift_x=0.8,
    occ_patch=None,
    keep_inside_mask=None,
    compact_box_len_scale=0.82,
    compact_box_wid_scale=0.92,
    short_tall_box_len_scale=0.94,
    short_tall_box_wid_scale=0.94,
    row_title=None, col_title=None, panel_color="#e8e8e8", border_color="#3f3f3f", border_width=1.0
):
    ax.set_facecolor(panel_color)
    bev_show = np.asarray(bev, dtype=np.float32)
    # Apply augmentation in visualization space (after display shift),
    # so erase coordinates match what user sees in the BEV panel.
    if occ_patch is not None:
        bev_show = apply_occ_patch(bev_show, *occ_patch)
    bev_show = shift_bev_x(bev_show, viz_joint_shift_x, dataset.bev_res)
    if keep_inside_mask is not None:
        bev_show = np.where(keep_inside_mask, bev_show, 0.0)
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
    if trailer_type == "temporary":
        box_len *= float(compact_box_len_scale)
        box_wid *= float(compact_box_wid_scale)
    elif trailer_type == "charger":
        box_len *= float(short_tall_box_len_scale)
        box_wid *= float(short_tall_box_wid_scale)

    gt_xy = box_corners(np.deg2rad(gt_deg), box_len, box_wid)
    pr_xy = box_corners(np.deg2rad(pr_deg), box_len, box_wid)
    gt_box = corners_to_pixels(gt_xy, x_min, y_min, res)
    pr_box = corners_to_pixels(pr_xy, x_min, y_min, res)
    ax.add_patch(plt.Polygon(gt_box, fill=False, edgecolor="lime", linewidth=1.25))
    ax.add_patch(plt.Polygon(pr_box, fill=False, edgecolor="red", linewidth=1.25))

    ax.add_patch(
        Rectangle(
            (0.012, 0.735), 0.24, 0.26,
            transform=ax.transAxes, facecolor="#5f5f5f", alpha=0.28, edgecolor="none", zorder=3
        )
    )
    ax.text(0.02, 0.98, f"GT {gt_deg:.1f}°", transform=ax.transAxes, ha="left", va="top",
            fontsize=11, color="lime", zorder=4)
    ax.text(0.02, 0.90, f"Pred {pr_deg:.1f}°", transform=ax.transAxes, ha="left", va="top",
            fontsize=11, color="red", zorder=4)
    ax.text(0.02, 0.82, f"Err {err_deg:+.1f}°", transform=ax.transAxes, ha="left", va="top",
            fontsize=11, color="white", zorder=4)

    if col_title is not None:
        ax.set_title(col_title, fontsize=16, pad=15)
    if row_title is not None:
        ax.text(-0.09, 0.5, row_title, transform=ax.transAxes, rotation=90,
                ha="center", va="center", fontsize=15, fontweight="bold")

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(border_color)
        spine.set_linewidth(border_width)


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    method_specs = [(m, None, None) for m in METHOD_ORDER]

    results = {m[0]: {} for m in method_specs}

    for trailer_type in TRAILER_ORDER:
        per_method = {}
        for method_name, _, _ in method_specs:
            cfg = MODEL_CFG_BY_TRAILER[trailer_type][method_name]
            per_method[method_name] = evaluate_model(
                cfg_path=cfg,
                ckpt_override=None,
                eval_trailer_type=trailer_type,
                device=device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )

        base_names = per_method[method_specs[0][0]]["frame_names"]
        for method_name, _, _ in method_specs[1:]:
            cur_names = per_method[method_name]["frame_names"]
            if len(cur_names) != len(base_names):
                raise RuntimeError(f"Sample count mismatch in trailer={trailer_type}, method={method_name}")
            for i in range(len(base_names)):
                if Path(cur_names[i]).name != Path(base_names[i]).name:
                    raise RuntimeError(f"Frame order mismatch in trailer={trailer_type}, method={method_name}")

        top_p = args.top_percent_short_compact if trailer_type in ("charger", "temporary") else args.top_percent
        idx = pick_index(
            np.abs(per_method["Naive BEV"]["err_deg"]),
            per_method["Naive BEV"]["gt_deg"],
            top_percent=top_p,
            min_abs_gt_deg=args.min_abs_gt_deg,
            seed=args.seed + TRAILER_ORDER.index(trailer_type),
        )
        fixed_spec = ""
        if trailer_type == "charger":
            fixed_spec = args.fixed_frame_charger
        elif trailer_type == "temporary":
            fixed_spec = args.fixed_frame_temporary
        elif trailer_type == "dummy":
            fixed_spec = args.fixed_frame_dummy

        if fixed_spec:
            seq_name, frame_name = parse_seq_frame(fixed_spec)
            fixed_idx = find_frame_index(
                per_method["Naive BEV"]["frame_names"],
                target_seq=seq_name,
                target_frame=frame_name,
            )
            if fixed_idx is not None:
                idx = int(fixed_idx)
            else:
                print(f"[WARN] Fixed frame not found for {trailer_type}: {fixed_spec}; using auto-selected sample.")

        for method_name, _, _ in method_specs:
            results[method_name][trailer_type] = {
                "res": per_method[method_name],
                "idx": idx,
            }

    fig, axes = plt.subplots(4, 3, figsize=(args.fig_w, args.fig_h))
    fig.patch.set_facecolor(args.fig_color)

    # Short--Tall: keep only points inside Naive BEV predicted box.
    naive_short = results["Naive BEV"]["charger"]["res"]
    naive_short_idx = results["Naive BEV"]["charger"]["idx"]
    naive_short_bev = naive_short["dataset"][naive_short_idx]["bev"][0].numpy()
    x_min_short, y_min_short = naive_short["dataset"].bev_x_range[0], naive_short["dataset"].bev_y_range[0]
    res_short = naive_short["dataset"].bev_res
    t_len = TRAILER_TYPES["charger"]["len"]
    t_wid = TRAILER_TYPES["charger"]["width"]
    norm_short = bool(naive_short["cfg"].get("train", {}).get("normalize_xy", False))
    ratio = max(t_len, t_wid) / 2.0
    box_len_short = (t_len / ratio if norm_short else t_len) * 1.4 * float(args.short_tall_box_len_scale)
    box_wid_short = (t_wid / ratio if norm_short else t_wid) * float(args.short_tall_box_wid_scale)
    pr_deg_short = float(naive_short["pred_deg"][naive_short_idx])
    pr_box_short = corners_to_pixels(
        box_corners(np.deg2rad(pr_deg_short), box_len_short, box_wid_short),
        x_min_short, y_min_short, res_short
    )
    short_tall_keep_mask = polygon_inside_mask(naive_short_bev.shape, pr_box_short)

    col_title_with_frame = {}
    for trailer_type in TRAILER_ORDER:
        col_title_with_frame[trailer_type] = f"{TRAILER_COL_TITLE[trailer_type]}\n{COL_SUBTITLE[trailer_type]}"

    for r, (method_name, _, _) in enumerate(method_specs):
        for c, trailer_type in enumerate(TRAILER_ORDER):
            pack = results[method_name][trailer_type]
            res = pack["res"]
            idx = pack["idx"]
            item = res["dataset"][idx]
            bev = item["bev"][0].numpy()
            plot_cell(
                axes[r, c],
                bev=bev,
                gt_deg=float(res["gt_deg"][idx]),
                pr_deg=float(res["pred_deg"][idx]),
                err_deg=float(res["err_deg"][idx]),
                dataset=res["dataset"],
                trailer_type=trailer_type,
                normalize_xy=bool(res["cfg"].get("train", {}).get("normalize_xy", False)),
                viz_joint_shift_x=(0.0 if method_name.startswith("Ours(") else args.viz_joint_shift_x),
                occ_patch=(
                    args.occ_cx_frac, args.occ_cy_frac, args.occ_w_frac, args.occ_h_frac
                ) if (args.occlude_short_tall and trailer_type == "charger") else None,
                keep_inside_mask=(short_tall_keep_mask if trailer_type == "charger" else None),
                compact_box_len_scale=args.compact_box_len_scale,
                compact_box_wid_scale=args.compact_box_wid_scale,
                short_tall_box_len_scale=args.short_tall_box_len_scale,
                short_tall_box_wid_scale=args.short_tall_box_wid_scale,
                row_title=method_name if c == 0 else None,
                col_title=col_title_with_frame[trailer_type] if r == 0 else None,
                panel_color=args.panel_color,
                border_color=args.border_color,
                border_width=args.border_width,
            )

    fig.subplots_adjust(left=0.08, right=0.995, top=0.955, bottom=0.035, hspace=0.0, wspace=0.05)
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(args.out, dpi=220, facecolor=args.fig_color)
    plt.close(fig)

    print(f"[INFO] Saved figure: {args.out}")
    for method_name, _, _ in method_specs:
        print(f"[INFO] {method_name}")
        for trailer_type in TRAILER_ORDER:
            idx = results[method_name][trailer_type]["idx"]
            fr = results[method_name][trailer_type]["res"]["frame_names"][idx]
            print(f"       {trailer_type}: {Path(fr).parent.name}/{Path(fr).name}")


if __name__ == "__main__":
    main()
