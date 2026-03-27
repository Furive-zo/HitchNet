#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Offline planning-stability proxy evaluation.

This script connects hitch-angle estimation to safety/operational stability metrics
without closed-loop simulation. It evaluates each method on the same logged test
sequences and reports:
  - Jackknife risk rate: P(|gamma| > gamma_safe), gamma_safe in {45, 60} deg
  - Minimum margin to jackknife: min_t (gamma_safe - |gamma(t)|)
  - Temporal consistency: jump rate P(|Delta gamma_t| > jump_thr)

All gamma values are wrapped angles from atan2(sin, cos), in degrees.
"""

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List, Tuple
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

# Allow running as `python scripts/planning_stability_eval.py`
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models import build_model
from utils.angle import wrap_deg_torch, wrap_rad_torch
from utils.collate import collate_fn
from utils.load_config import load_config
from utils.load_dataset import HitchDataset


SCENARIO_GROUPS = {
    "low_curvature": {"highway_5", "urban_5"},
    "high_curvature": {"reverse_4", "scurve_5"},
    "elevated": {"updown_9"},
}

SCENARIO_LABELS = {
    "low_curvature": "Low Curvature",
    "high_curvature": "High Curvature",
    "elevated": "Elevated",
}

DEFAULT_MODELS = [
    ("GT", "-", "-", "#111111"),
    ("Naive BEV", "configs/experiments/dummy_bev_resnet_regression.yaml", "-", "#1D4E89"),
    ("Aligned BEV", "configs/experiments/dummy_bev_resnet_regression_norm.yaml", "-", "#2C7FB8"),
    ("Proposed", "configs/experiments/dummy_bev_resnet_regression_norm_aug.yaml", "-", "#6BAED6"),
]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"],
    "font.size": 12,
})


def parse_args():
    p = argparse.ArgumentParser(description="Offline planning stability evaluation")
    p.add_argument(
        "--models",
        type=str,
        nargs="*",
        default=None,
        help=(
            "Model spec list: name|config|ckpt|color . "
            "ckpt can be '-' (auto best.pth) or comma-separated ckpts for seed aggregation."
        ),
    )
    p.add_argument(
        "--eval_trailer_type",
        type=str,
        choices=["charger", "dummy", "temporary"],
        required=True,
        help="Trailer config to evaluate on test split.",
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--jump_thr_deg", type=float, default=5.0)
    p.add_argument("--lead_window", type=int, default=20, help="Look-back window (frames) for lead-time-to-risk.")
    p.add_argument("--risk_thrs_deg", type=float, nargs="+", default=[35.0])
    p.add_argument(
        "--scenario_plot_metric",
        type=str,
        default="miss_rate_35",
        help="Metric key for scenario-wise plot. e.g., miss_rate_35, false_alarm_rate_35, lead_time_to_risk_35, jump_rate_err",
    )
    p.add_argument("--out_dir", type=str, default="results/planning_stability")
    return p.parse_args()


def parse_models(model_specs):
    if not model_specs:
        return DEFAULT_MODELS
    models = []
    for spec in model_specs:
        parts = spec.split("|")
        if len(parts) != 4:
            raise ValueError(f"Invalid model spec: {spec}")
        name, cfg, ckpt, color = [x.strip() for x in parts]
        models.append((name, cfg, ckpt, color))
    return models


def move_batch_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
    return out


def get_group_name(seq_name: str) -> str:
    for group, seqs in SCENARIO_GROUPS.items():
        if seq_name in seqs:
            return group
    return "other"


def resolve_eval_dataset(cfg_path: str, eval_trailer_type: str) -> Dict:
    cfg_dir = os.path.dirname(cfg_path)
    ds_rel = f"datasets/{eval_trailer_type}.yaml"
    ds_path = os.path.normpath(os.path.join(cfg_dir, "..", ds_rel))
    with open(ds_path, "r") as f:
        ds_cfg = yaml.safe_load(f)
    return ds_cfg.get("dataset", {})


def build_dataset_loader(cfg, cfg_path, eval_trailer_type, batch_size_override, num_workers_override):
    dset_cfg = resolve_eval_dataset(cfg_path, eval_trailer_type)
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
        stats_path = train_cfg.get(
            "dlog_stats_path",
            os.path.join("ckpts", exp_name, f"dlog_range_stats_b{bin_size}.pt"),
        )
        if os.path.exists(stats_path):
            stats = torch.load(stats_path, map_location="cpu", weights_only=False)
            dataset.dlog_range_stats = stats
            dataset._build_range_bin_idx()

    num_workers = num_workers_override if num_workers_override is not None else int(dset_cfg.get("num_workers", 4))
    batch_size = batch_size_override if batch_size_override is not None else int(train_cfg.get("batch_size", 8))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    return dataset, loader


def infer_gamma_deg(method_name, cfg_path, ckpt_path, device, eval_trailer_type, batch_size, num_workers):
    cfg = load_config(cfg_path)
    dataset, loader = build_dataset_loader(cfg, cfg_path, eval_trailer_type, batch_size, num_workers)

    if method_name.upper() == "GT":
        use_gt_direct = True
        model = None
    else:
        use_gt_direct = False
        model_cfg = cfg["model"]
        model = build_model(model_cfg).to(device)
        if model_cfg.get("name") not in ("rule_based", "rule_based_pca", "rule_based_ols", "rule_based_mle"):
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"[{method_name}] checkpoint not found: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt, strict=True)
        model.eval()

    frame_rows = []
    idx_global = 0

    with torch.no_grad():
        for batch in loader:
            bsz = batch["gt"].shape[0]
            batch_dev = move_batch_to_device(batch, device)

            if use_gt_direct:
                pred = batch_dev["gt"]
            else:
                pred = model(batch_dev)
            gt = batch_dev["gt"]

            theta_p = torch.atan2(pred[:, 1], pred[:, 0])
            theta_g = torch.atan2(gt[:, 1], gt[:, 0])
            err_deg = wrap_rad_torch(theta_p - theta_g) * 180.0 / np.pi

            pred_deg = theta_p.detach().cpu() * 180.0 / np.pi
            gt_deg = theta_g.detach().cpu() * 180.0 / np.pi
            err_deg = err_deg.detach().cpu()

            for i in range(bsz):
                fr = dataset.frame_dirs[idx_global + i]
                seq_name = Path(fr).parent.name
                frame_name = Path(fr).name
                frame_idx = -1
                if frame_name.startswith("frame_"):
                    try:
                        frame_idx = int(frame_name.split("_")[-1])
                    except ValueError:
                        frame_idx = -1
                frame_rows.append(
                    {
                        "seq": seq_name,
                        "frame_idx": frame_idx,
                        "scenario": get_group_name(seq_name),
                        "pred_deg": float(pred_deg[i].item()),
                        "gt_deg": float(gt_deg[i].item()),
                        "err_deg": float(err_deg[i].item()),
                    }
                )
            idx_global += bsz

    return frame_rows


def compute_metrics(frame_rows: List[Dict], risk_thrs_deg: List[float], jump_thr_deg: float, lead_window: int):
    by_group = {"overall": frame_rows}
    for g in SCENARIO_GROUPS.keys():
        by_group[g] = [r for r in frame_rows if r["scenario"] == g]

    def _metrics(rows):
        if not rows:
            out = {
                "n_frames": 0,
                "jump_rate": np.nan,  # backward-compat alias for pred jump
                "mean_abs_jump": np.nan,  # backward-compat alias for pred jump magnitude
                "jump_rate_pred": np.nan,
                "mean_abs_jump_pred": np.nan,
                "jump_rate_err": np.nan,
                "mean_abs_jump_err": np.nan,
            }
            for thr in risk_thrs_deg:
                out[f"risk_rate_{int(thr)}"] = np.nan
                out[f"min_margin_{int(thr)}"] = np.nan
                out[f"p1_margin_{int(thr)}"] = np.nan
                out[f"p5_margin_{int(thr)}"] = np.nan
                out[f"miss_rate_{int(thr)}"] = np.nan
                out[f"false_alarm_rate_{int(thr)}"] = np.nan
                out[f"margin_error_{int(thr)}"] = np.nan
                out[f"abs_margin_error_{int(thr)}"] = np.nan
                out[f"lead_time_to_risk_{int(thr)}"] = np.nan
                out[f"lead_detect_rate_{int(thr)}"] = np.nan
            return out

        pred = np.array([r["pred_deg"] for r in rows], dtype=np.float32)
        gt = np.array([r["gt_deg"] for r in rows], dtype=np.float32)
        err = np.array([r["err_deg"] for r in rows], dtype=np.float32)
        pred_abs = np.abs(pred)
        gt_abs = np.abs(gt)
        out = {"n_frames": int(len(rows))}
        for thr in risk_thrs_deg:
            thr_f = float(thr)
            margin = thr_f - pred_abs
            out[f"risk_rate_{int(thr)}"] = float(np.mean(pred_abs > float(thr)))
            out[f"min_margin_{int(thr)}"] = float(np.min(margin))
            out[f"p1_margin_{int(thr)}"] = float(np.percentile(margin, 1))
            out[f"p5_margin_{int(thr)}"] = float(np.percentile(margin, 5))
            gt_risk = gt_abs > thr_f
            pred_risk = pred_abs > thr_f
            out[f"miss_rate_{int(thr)}"] = float(np.mean(gt_risk & (~pred_risk)))
            out[f"false_alarm_rate_{int(thr)}"] = float(np.mean((~gt_risk) & pred_risk))
            margin_err = gt_abs - pred_abs  # (thr-|pred|) - (thr-|gt|) with sign flipped for readability
            out[f"margin_error_{int(thr)}"] = float(np.mean(margin_err))
            out[f"abs_margin_error_{int(thr)}"] = float(np.mean(np.abs(margin_err)))

            # Lead-time to risk onset: earliest predicted-risk within look-back window.
            seq_to_pairs = {}
            for r in rows:
                seq_to_pairs.setdefault(r["seq"], []).append((r.get("frame_idx", -1), abs(r["gt_deg"]), abs(r["pred_deg"])))
            leads = []
            n_onsets = 0
            n_detect = 0
            for vals in seq_to_pairs.values():
                vals_sorted = sorted(vals, key=lambda x: x[0])
                g = np.array([v[1] for v in vals_sorted], dtype=np.float32)
                p = np.array([v[2] for v in vals_sorted], dtype=np.float32)
                g_risk = g > thr_f
                p_risk = p > thr_f
                onset_idx = np.where(g_risk & np.concatenate(([True], ~g_risk[:-1])))[0]
                for t in onset_idx.tolist():
                    n_onsets += 1
                    s = max(0, t - int(lead_window))
                    cand = np.where(p_risk[s:t + 1])[0]
                    if cand.size > 0:
                        k = s + int(cand[0])  # earliest detection in look-back window
                        leads.append(float(t - k))
                        n_detect += 1
            out[f"lead_time_to_risk_{int(thr)}"] = float(np.mean(leads)) if len(leads) > 0 else np.nan
            out[f"lead_detect_rate_{int(thr)}"] = float(n_detect / n_onsets) if n_onsets > 0 else np.nan

        seq_to_vals = {}
        for r in rows:
            seq_to_vals.setdefault(r["seq"], []).append((r.get("frame_idx", -1), r["pred_deg"], r["err_deg"]))
        jumps_pred = []
        jumps_err = []
        for vals in seq_to_vals.values():
            if len(vals) < 2:
                continue
            vals_sorted = sorted(vals, key=lambda x: x[0])
            vp = torch.tensor([x[1] for x in vals_sorted], dtype=torch.float32)
            ve = torch.tensor([x[2] for x in vals_sorted], dtype=torch.float32)
            dp = torch.abs(wrap_deg_torch(vp[1:] - vp[:-1]))
            de = torch.abs(wrap_deg_torch(ve[1:] - ve[:-1]))
            jumps_pred.append(dp)
            jumps_err.append(de)
        if len(jumps_pred) == 0:
            out["jump_rate_pred"] = np.nan
            out["mean_abs_jump_pred"] = np.nan
            out["jump_rate_err"] = np.nan
            out["mean_abs_jump_err"] = np.nan
            out["jump_rate"] = np.nan
            out["mean_abs_jump"] = np.nan
        else:
            jp = torch.cat(jumps_pred, dim=0).numpy()
            je = torch.cat(jumps_err, dim=0).numpy()
            out["jump_rate_pred"] = float(np.mean(jp > float(jump_thr_deg)))
            out["mean_abs_jump_pred"] = float(np.mean(jp))
            out["jump_rate_err"] = float(np.mean(je > float(jump_thr_deg)))
            out["mean_abs_jump_err"] = float(np.mean(je))
            # backward-compat aliases
            out["jump_rate"] = out["jump_rate_pred"]
            out["mean_abs_jump"] = out["mean_abs_jump_pred"]
        return out

    return {k: _metrics(v) for k, v in by_group.items()}


def summarize_runs(run_metrics: List[Dict], groups: List[str], risk_thrs_deg: List[float]):
    keys = [
        "jump_rate", "mean_abs_jump",
        "jump_rate_pred", "mean_abs_jump_pred",
        "jump_rate_err", "mean_abs_jump_err",
    ]
    for thr in risk_thrs_deg:
        keys.append(f"risk_rate_{int(thr)}")
        keys.append(f"min_margin_{int(thr)}")
        keys.append(f"p1_margin_{int(thr)}")
        keys.append(f"p5_margin_{int(thr)}")
        keys.append(f"miss_rate_{int(thr)}")
        keys.append(f"false_alarm_rate_{int(thr)}")
        keys.append(f"margin_error_{int(thr)}")
        keys.append(f"abs_margin_error_{int(thr)}")
        keys.append(f"lead_time_to_risk_{int(thr)}")
        keys.append(f"lead_detect_rate_{int(thr)}")

    summary = {}
    for g in groups:
        summary[g] = {"n_frames": int(np.mean([m[g]["n_frames"] for m in run_metrics]))}
        for k in keys:
            vals = np.array([m[g][k] for m in run_metrics], dtype=np.float32)
            summary[g][f"{k}_mean"] = float(np.nanmean(vals))
            summary[g][f"{k}_std"] = float(np.nanstd(vals))
    return summary


def auto_ckpt_from_config(cfg_path: str):
    cfg = load_config(cfg_path)
    exp_name = cfg.get("experiment", {}).get("name", Path(cfg_path).stem)
    return os.path.join("ckpts", exp_name, "best.pth")


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def metric_axis_label(metric_key: str):
    if metric_key.startswith("risk_rate_"):
        thr = metric_key.split("_")[-1]
        return f"Jackknife Risk Rate (|$\\gamma$| > {thr}°, %)"
    if metric_key == "jump_rate":
        return "Temporal Jump Rate (%)"
    if metric_key == "jump_rate_pred":
        return "Prediction Jump Rate (%)"
    if metric_key == "jump_rate_err":
        return "Error Jump Rate (%)"
    if metric_key == "mean_abs_jump":
        return "Mean |Δ$\\gamma$| (°/frame)"
    if metric_key == "mean_abs_jump_pred":
        return "Mean |Δpred| (°/frame)"
    if metric_key == "mean_abs_jump_err":
        return "Mean |Δerr| (°/frame)"
    if metric_key.startswith("miss_rate_"):
        thr = metric_key.split("_")[-1]
        return f"Miss Rate (|gt| > {thr}° and |pred| ≤ {thr}°, %)"
    if metric_key.startswith("false_alarm_rate_"):
        thr = metric_key.split("_")[-1]
        return f"False Alarm Rate (|gt| ≤ {thr}° and |pred| > {thr}°, %)"
    if metric_key.startswith("lead_time_to_risk_"):
        thr = metric_key.split("_")[-1]
        return f"Lead Time to Risk @ {thr}° (frames)"
    if metric_key.startswith("lead_detect_rate_"):
        thr = metric_key.split("_")[-1]
        return f"Lead Detection Rate @ {thr}° (%)"
    if metric_key.startswith("margin_error_"):
        thr = metric_key.split("_")[-1]
        return f"Mean Margin Error to {thr}° (°)"
    if metric_key.startswith("abs_margin_error_"):
        thr = metric_key.split("_")[-1]
        return f"Mean |Margin Error| to {thr}° (°)"
    if metric_key.startswith("min_margin_"):
        thr = metric_key.split("_")[-1]
        return f"Minimum Safety Margin to {thr}° (°)"
    if metric_key.startswith("p1_margin_"):
        thr = metric_key.split("_")[-1]
        return f"1st-Percentile Safety Margin to {thr}° (°)"
    if metric_key.startswith("p5_margin_"):
        thr = metric_key.split("_")[-1]
        return f"5th-Percentile Safety Margin to {thr}° (°)"
    return metric_key


def metric_scale(metric_key: str):
    if (
        metric_key.startswith("risk_rate_")
        or metric_key.startswith("miss_rate_")
        or metric_key.startswith("false_alarm_rate_")
        or metric_key.startswith("lead_detect_rate_")
        or metric_key == "jump_rate"
        or metric_key == "jump_rate_pred"
        or metric_key == "jump_rate_err"
    ):
        return 100.0
    return 1.0


def main():
    args = parse_args()
    models = parse_models(args.models)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ensure_dir(args.out_dir)
    groups = ["overall"] + list(SCENARIO_GROUPS.keys())

    per_run_rows = []
    summary_rows = []
    plot_names = []
    plot_vals = []
    plot_errs = []
    plot_colors = []

    for name, cfg_path, ckpt_spec, color in models:
        if cfg_path == "-":
            # GT uses config from first non-GT model for dataset/inference path.
            fallback = next((m for m in models if m[1] != "-"), None)
            if fallback is None:
                raise ValueError("At least one non-GT model config is required.")
            cfg_path = fallback[1]

        if ckpt_spec == "-":
            ckpt_list = [auto_ckpt_from_config(cfg_path)]
        else:
            ckpt_list = [x.strip() for x in ckpt_spec.split(",") if x.strip()]

        run_metrics = []
        for run_idx, ckpt_path in enumerate(ckpt_list):
            rows = infer_gamma_deg(
                method_name=name,
                cfg_path=cfg_path,
                ckpt_path=ckpt_path,
                device=device,
                eval_trailer_type=args.eval_trailer_type,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
            metrics = compute_metrics(rows, args.risk_thrs_deg, args.jump_thr_deg, args.lead_window)
            run_metrics.append(metrics)

            for g in groups:
                item = {
                    "method": name,
                    "run": run_idx,
                    "trailer_type": args.eval_trailer_type,
                    "group": g,
                    "n_frames": metrics[g]["n_frames"],
                    "jump_rate": metrics[g]["jump_rate"],
                    "mean_abs_jump": metrics[g]["mean_abs_jump"],
                    "jump_rate_pred": metrics[g]["jump_rate_pred"],
                    "mean_abs_jump_pred": metrics[g]["mean_abs_jump_pred"],
                    "jump_rate_err": metrics[g]["jump_rate_err"],
                    "mean_abs_jump_err": metrics[g]["mean_abs_jump_err"],
                }
                for thr in args.risk_thrs_deg:
                    item[f"risk_rate_{int(thr)}"] = metrics[g][f"risk_rate_{int(thr)}"]
                    item[f"min_margin_{int(thr)}"] = metrics[g][f"min_margin_{int(thr)}"]
                    item[f"p1_margin_{int(thr)}"] = metrics[g][f"p1_margin_{int(thr)}"]
                    item[f"p5_margin_{int(thr)}"] = metrics[g][f"p5_margin_{int(thr)}"]
                    item[f"miss_rate_{int(thr)}"] = metrics[g][f"miss_rate_{int(thr)}"]
                    item[f"false_alarm_rate_{int(thr)}"] = metrics[g][f"false_alarm_rate_{int(thr)}"]
                    item[f"margin_error_{int(thr)}"] = metrics[g][f"margin_error_{int(thr)}"]
                    item[f"abs_margin_error_{int(thr)}"] = metrics[g][f"abs_margin_error_{int(thr)}"]
                    item[f"lead_time_to_risk_{int(thr)}"] = metrics[g][f"lead_time_to_risk_{int(thr)}"]
                    item[f"lead_detect_rate_{int(thr)}"] = metrics[g][f"lead_detect_rate_{int(thr)}"]
                per_run_rows.append(item)

        summary = summarize_runs(run_metrics, groups, args.risk_thrs_deg)
        for g in groups:
            row = {
                "method": name,
                "n_runs": len(run_metrics),
                "trailer_type": args.eval_trailer_type,
                "group": g,
                "n_frames": summary[g]["n_frames"],
            }
            for k, v in summary[g].items():
                if k != "n_frames":
                    row[k] = v
            summary_rows.append(row)

        primary_thr = 45 if any(int(t) == 45 for t in args.risk_thrs_deg) else int(args.risk_thrs_deg[0])
        plot_names.append(name)
        plot_vals.append(summary["overall"][f"risk_rate_{primary_thr}_mean"])
        plot_errs.append(summary["overall"][f"risk_rate_{primary_thr}_std"])
        plot_colors.append(color)

    per_run_csv = os.path.join(args.out_dir, f"planning_stability_per_run_{args.eval_trailer_type}.csv")
    summary_csv = os.path.join(args.out_dir, f"planning_stability_summary_{args.eval_trailer_type}.csv")

    with open(per_run_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(per_run_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_run_rows)

    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    fig_path = os.path.join(args.out_dir, f"planning_risk{primary_thr}_{args.eval_trailer_type}.png")
    xs = np.arange(len(plot_names))
    plt.figure(figsize=(8.0, 4.2))
    risk_scale = 100.0
    plot_vals_scaled = [v * risk_scale for v in plot_vals]
    plot_errs_scaled = [e * risk_scale for e in plot_errs]
    bars = plt.bar(
        xs,
        plot_vals_scaled,
        yerr=plot_errs_scaled,
        capsize=4.0,
        width=0.58,
        color=plot_colors,
        edgecolor="#222222",
    )
    plt.xticks(xs, plot_names, rotation=20, ha="right")
    plt.ylabel(f"Jackknife Risk Rate (|$\\gamma$| > {primary_thr}°, %)")
    ax = plt.gca()
    for b, y in zip(bars, plot_vals_scaled):
        if np.isfinite(y):
            ax.text(
                b.get_x() + b.get_width() / 2.0,
                y,
                f"{y:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.20),
        ncol=max(1, min(len(plot_names), 5)),
    )
    plt.tight_layout()
    plt.savefig(fig_path, dpi=180)
    plt.close()

    # Scenario-wise grouped plot (recommended for paper/appendix)
    scenario_metric = args.scenario_plot_metric
    scenario_groups = list(SCENARIO_GROUPS.keys())
    scenario_scale = metric_scale(scenario_metric)
    scen_x = np.arange(len(scenario_groups))
    n_methods = max(1, len(plot_names))
    total_w = 0.72
    bw = total_w / n_methods
    plt.figure(figsize=(9.2, 4.4))
    for mi, method in enumerate(plot_names):
        vals = []
        errs = []
        for sg in scenario_groups:
            row = next((r for r in summary_rows if r["method"] == method and r["group"] == sg), None)
            if row is None:
                vals.append(np.nan)
                errs.append(np.nan)
            else:
                vals.append(row.get(f"{scenario_metric}_mean", np.nan) * scenario_scale)
                errs.append(row.get(f"{scenario_metric}_std", np.nan) * scenario_scale)
        offset = -total_w / 2 + (mi + 0.5) * bw
        bars = plt.bar(
            scen_x + offset,
            vals,
            yerr=errs,
            capsize=3.0,
            width=bw * 0.82,
            label=method,
            color=plot_colors[mi],
            edgecolor="#222222",
        )
        ax = plt.gca()
        for b, y in zip(bars, vals):
            if np.isfinite(y):
                ax.text(
                    b.get_x() + b.get_width() / 2.0,
                    y,
                    f"{y:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
    plt.xticks(scen_x, [SCENARIO_LABELS.get(s, s) for s in scenario_groups])
    plt.ylabel(metric_axis_label(scenario_metric))
    ax = plt.gca()
    ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.22),
        ncol=max(1, min(len(plot_names), 5)),
    )
    plt.tight_layout()
    scen_fig_path = os.path.join(
        args.out_dir, f"planning_{scenario_metric}_by_scenario_{args.eval_trailer_type}.png"
    )
    plt.savefig(scen_fig_path, dpi=180)
    plt.close()

    print(f"[INFO] Saved per-run CSV: {per_run_csv}")
    print(f"[INFO] Saved summary CSV: {summary_csv}")
    print(f"[INFO] Saved figure: {fig_path}")
    print(f"[INFO] Saved scenario figure: {scen_fig_path}")

    print(f"\n[LaTeX rows | overall | risk threshold={primary_thr}deg]")
    for row in summary_rows:
        if row["group"] != "overall":
            continue
        rr = f"{row.get(f'risk_rate_{primary_thr}_mean', np.nan):.4f}\\pm{row.get(f'risk_rate_{primary_thr}_std', np.nan):.4f}"
        mr = f"{row.get(f'min_margin_{primary_thr}_mean', np.nan):.2f}\\pm{row.get(f'min_margin_{primary_thr}_std', np.nan):.2f}"
        p5 = f"{row.get(f'p5_margin_{primary_thr}_mean', np.nan):.2f}\\pm{row.get(f'p5_margin_{primary_thr}_std', np.nan):.2f}"
        jr = f"{row.get('jump_rate_mean', np.nan):.4f}\\pm{row.get('jump_rate_std', np.nan):.4f}"
        print(f"{row['method']} & {rr} & {mr} & {p5} & {jr} \\\\")


if __name__ == "__main__":
    main()
