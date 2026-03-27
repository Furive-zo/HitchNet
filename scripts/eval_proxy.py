#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""CLI entrypoint for curvature-conditioned trailer deviation proxy evaluation."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List

import numpy as np

from scripts.proxy_eval.io import (
    expand_input_glob,
    load_log_file,
    load_trailer_rules,
    match_trailer_params,
    parse_colmap,
    parse_rotation_matrix,
    override_yaw_rate_from_imu_csv,
    override_yaw_rate_from_vehicle_imu_json,
)
from scripts.proxy_eval.kinematics import compute_proxy_timeseries
from scripts.proxy_eval.metrics import curvature_binned_stats, overall_stats
from scripts.proxy_eval.plotting import plot_exceedance_vs_kappa, plot_p95_vs_kappa


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Curvature-conditioned trailer deviation proxy evaluator")
    p.add_argument("--input_glob", type=str, required=True, help='Input file glob, e.g., "logs/*.csv"')
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--L2", type=float, default=6.2, help="Hitch->trailer axle distance (m)")
    p.add_argument("--eps", type=float, default=0.0, help="Rear axle->hitch offset (m)")
    p.add_argument("--config_json", type=str, default=None, help="Optional trailer param rule config.")
    p.add_argument("--v_min", type=float, default=1.0)
    p.add_argument("--gamma_unit", type=str, choices=["rad", "deg"], default="rad")
    p.add_argument("--gamma_sign", type=int, choices=[-1, 1], default=1)
    p.add_argument(
        "--yaw_rate_source",
        type=str,
        choices=["csv", "imu_csv", "vehicle_imu_json"],
        default="csv",
        help="yaw_rate source: use CSV column directly, derive from imu columns, or derive from per-frame vehicle_imu.json",
    )
    p.add_argument(
        "--imu_rotation",
        type=str,
        default="0,0,-1,0,-1,0,-1,0,0",
        help="R_FLU_from_RT_vehicle as 9 comma-separated values.",
    )
    p.add_argument(
        "--imu_frame_base_dir",
        type=str,
        default="",
        help="Base dir for relative frame_dir values when yaw_rate_source=vehicle_imu_json",
    )
    p.add_argument(
        "--colmap",
        type=str,
        default="",
        help="Column map override: canon:actual,... e.g., v:speed,yaw_rate:gyro_z,gamma_gt:gt,gamma_pred:pred",
    )
    p.add_argument("--method", type=str, default=None, help="Fixed method name for all inputs.")
    p.add_argument("--method_from", type=str, choices=["stem", "parent"], default="parent")
    p.add_argument(
        "--kappa_edges",
        type=str,
        default="0.0,0.02,0.04,0.06,0.08,0.10,0.12,0.14",
        help="Comma-separated |kappa| bin edges.",
    )
    p.add_argument("--min_bin_n", type=int, default=50)
    p.add_argument("--thr_a", type=float, default=0.2, help="Exceedance threshold A [m]")
    p.add_argument("--thr_b", type=float, default=0.3, help="Exceedance threshold B [m]")
    p.add_argument("--save_traj", action="store_true", help="Save per-sequence reconstructed trajectories.")
    return p.parse_args()


def parse_edges(spec: str) -> np.ndarray:
    vals = [float(x.strip()) for x in spec.split(",") if x.strip()]
    if len(vals) < 2:
        raise ValueError("--kappa_edges needs at least 2 values.")
    arr = np.asarray(vals, dtype=np.float64)
    if np.any(np.diff(arr) <= 0):
        raise ValueError("--kappa_edges must be strictly increasing.")
    return arr


def infer_method_name(path: str, method_from: str) -> str:
    p = Path(path)
    if method_from == "parent":
        name = p.parent.name
        if name:
            return name
    return p.stem


def write_csv(path: str, rows: List[Dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    fieldnames = sorted(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def write_readme(out_dir: str, args: argparse.Namespace, n_files: int) -> None:
    text = f"""# Curvature-Conditioned Trailer Deviation Proxy

- Input files: `{n_files}`
- `v_min`: {args.v_min}
- `gamma_unit`: {args.gamma_unit}
- `gamma_sign`: {args.gamma_sign}
- `yaw_rate_source`: {args.yaw_rate_source}
- `imu_rotation`: {args.imu_rotation}
- `L2`: {args.L2}
- `eps`: {args.eps}
- `kappa_edges`: {args.kappa_edges}
- `min_bin_n`: {args.min_bin_n}
- exceed thresholds: {args.thr_a} m, {args.thr_b} m

## Outputs
- `summary_overall.csv`
- `summary_by_kappa_bin.csv`
- `figures/p95_abs_ey_vs_kappa.png`
- `figures/exceed_{args.thr_a:.1f}m_vs_kappa.png`
- `figures/exceed_{args.thr_b:.1f}m_vs_kappa.png`
- `traj/*.csv` (if `--save_traj`)
"""
    with open(os.path.join(out_dir, "README.md"), "w") as f:
        f.write(text)


def main() -> None:
    args = parse_args()
    files = expand_input_glob(args.input_glob)
    colmap = parse_colmap(args.colmap)
    rule_cfg = load_trailer_rules(args.config_json)
    kappa_edges = parse_edges(args.kappa_edges)
    R_flu_from_rt = parse_rotation_matrix(args.imu_rotation)

    out_dir = args.out_dir
    fig_dir = os.path.join(out_dir, "figures")
    traj_dir = os.path.join(out_dir, "traj")
    os.makedirs(fig_dir, exist_ok=True)
    if args.save_traj:
        os.makedirs(traj_dir, exist_ok=True)

    overall_rows: List[Dict] = []
    by_bin_rows: List[Dict] = []

    for fp in files:
        method = args.method if args.method else infer_method_name(fp, args.method_from)
        seq = Path(fp).stem
        params = match_trailer_params(fp, args.L2, args.eps, rule_cfg)
        log = load_log_file(
            fp,
            colmap=colmap,
            gamma_unit=args.gamma_unit,
            require_yaw_rate=(args.yaw_rate_source == "csv"),
        )
        if args.yaw_rate_source == "imu_csv":
            override_yaw_rate_from_imu_csv(log, raw_path=fp, colmap=colmap, R_flu_from_rt=R_flu_from_rt)
        elif args.yaw_rate_source == "vehicle_imu_json":
            override_yaw_rate_from_vehicle_imu_json(
                log,
                raw_path=fp,
                colmap=colmap,
                R_flu_from_rt=R_flu_from_rt,
                imu_frame_base_dir=args.imu_frame_base_dir,
            )
        ts = compute_proxy_timeseries(
            log=log,
            L2=params.L2,
            eps=params.eps,
            gamma_sign=args.gamma_sign,
            v_min=args.v_min,
        )
        m = ts["valid_mask"]
        ey = ts["ey"][m]
        kappa = ts["kappa"][m]

        o = overall_stats(ey, thr_a=args.thr_a, thr_b=args.thr_b)
        o.update(
            {
                "method": method,
                "sequence": seq,
                "file": fp,
                "L2": params.L2,
                "eps": params.eps,
                "rule": params.matched_rule,
                "n_total": int(ts["ey"].size),
                "n_valid": int(np.sum(m)),
            }
        )
        overall_rows.append(o)

        bstats = curvature_binned_stats(
            ey=ey,
            kappa=kappa,
            bin_edges=kappa_edges,
            min_n_per_bin=args.min_bin_n,
            thr_a=args.thr_a,
            thr_b=args.thr_b,
        )
        for r in bstats:
            r.update(
                {
                    "method": method,
                    "sequence": seq,
                    "file": fp,
                    "L2": params.L2,
                    "eps": params.eps,
                    "rule": params.matched_rule,
                }
            )
            by_bin_rows.append(r)

        if args.save_traj:
            out_traj = os.path.join(traj_dir, f"{method}__{seq}.csv")
            rows = []
            for i in range(ts["t"].shape[0]):
                rows.append(
                    {
                        "t": float(ts["t"][i]),
                        "x": float(ts["x"][i]),
                        "y": float(ts["y"][i]),
                        "phi": float(ts["phi"][i]),
                        "v": float(ts["v"][i]),
                        "yaw_rate": float(ts["yaw_rate"][i]),
                        "kappa": float(ts["kappa"][i]),
                        "gamma_gt": float(ts["gamma_gt"][i]),
                        "gamma_pred": float(ts["gamma_pred"][i]),
                        "xtr_gt": float(ts["xtr_gt"][i]),
                        "ytr_gt": float(ts["ytr_gt"][i]),
                        "xtr_pred": float(ts["xtr_pred"][i]),
                        "ytr_pred": float(ts["ytr_pred"][i]),
                        "ey": float(ts["ey"][i]),
                        "valid": int(bool(ts["valid_mask"][i])),
                    }
                )
            write_csv(out_traj, rows)

    # Add method-level aggregate rows.
    method_names = sorted({r["method"] for r in overall_rows})
    for method in method_names:
        rows = [r for r in overall_rows if r["method"] == method]
        ey_all = []
        for r in rows:
            # Re-read per sequence via stored file to avoid storing huge arrays in memory.
            fp = r["file"]
            params = match_trailer_params(fp, args.L2, args.eps, rule_cfg)
            log = load_log_file(fp, colmap=colmap, gamma_unit=args.gamma_unit)
            ts = compute_proxy_timeseries(log, params.L2, params.eps, args.gamma_sign, args.v_min)
            ey_all.append(ts["ey"][ts["valid_mask"]])
        ey_cat = np.concatenate(ey_all, axis=0) if ey_all else np.array([], dtype=np.float64)
        agg = overall_stats(ey_cat, thr_a=args.thr_a, thr_b=args.thr_b)
        agg.update(
            {
                "method": method,
                "sequence": "__ALL__",
                "file": "__ALL__",
                "L2": np.nan,
                "eps": np.nan,
                "rule": "aggregate",
                "n_total": int(sum(r["n_total"] for r in rows)),
                "n_valid": int(sum(r["n_valid"] for r in rows)),
            }
        )
        overall_rows.append(agg)

    # Method-level aggregation for by-bin rows.
    for method in method_names:
        src = [r for r in by_bin_rows if r["method"] == method]
        for lo, hi in zip(kappa_edges[:-1], kappa_edges[1:]):
            br = [r for r in src if np.isclose(r["kappa_bin_lo"], lo) and np.isclose(r["kappa_bin_hi"], hi)]
            n_total = int(np.sum([int(r["n"]) for r in br])) if br else 0
            row = {
                "method": method,
                "sequence": "__ALL__",
                "file": "__ALL__",
                "kappa_bin_lo": float(lo),
                "kappa_bin_hi": float(hi),
                "kappa_bin_center": 0.5 * (float(lo) + float(hi)),
                "n": n_total,
                f"exceed_rate_{args.thr_a:.1f}m": np.nan,
                f"exceed_rate_{args.thr_b:.1f}m": np.nan,
                "p95_abs_ey": np.nan,
                "L2": np.nan,
                "eps": np.nan,
                "rule": "aggregate",
            }
            if n_total >= args.min_bin_n and br:
                vals_p95 = np.array([r["p95_abs_ey"] for r in br if np.isfinite(r["p95_abs_ey"])], dtype=float)
                vals_a = np.array([r[f"exceed_rate_{args.thr_a:.1f}m"] for r in br if np.isfinite(r[f"exceed_rate_{args.thr_a:.1f}m"])], dtype=float)
                vals_b = np.array([r[f"exceed_rate_{args.thr_b:.1f}m"] for r in br if np.isfinite(r[f"exceed_rate_{args.thr_b:.1f}m"])], dtype=float)
                if vals_p95.size > 0:
                    row["p95_abs_ey"] = float(np.mean(vals_p95))
                if vals_a.size > 0:
                    row[f"exceed_rate_{args.thr_a:.1f}m"] = float(np.mean(vals_a))
                if vals_b.size > 0:
                    row[f"exceed_rate_{args.thr_b:.1f}m"] = float(np.mean(vals_b))
            by_bin_rows.append(row)

    overall_csv = os.path.join(out_dir, "summary_overall.csv")
    by_bin_csv = os.path.join(out_dir, "summary_by_kappa_bin.csv")
    write_csv(overall_csv, overall_rows)
    write_csv(by_bin_csv, by_bin_rows)

    # Plot aggregated (__ALL__) rows only.
    agg_by_bin = [r for r in by_bin_rows if r["sequence"] == "__ALL__"]
    plot_p95_vs_kappa(agg_by_bin, os.path.join(fig_dir, "p95_abs_ey_vs_kappa.png"))
    plot_exceedance_vs_kappa(agg_by_bin, os.path.join(fig_dir, f"exceed_{args.thr_a:.1f}m_vs_kappa.png"), args.thr_a)
    plot_exceedance_vs_kappa(agg_by_bin, os.path.join(fig_dir, f"exceed_{args.thr_b:.1f}m_vs_kappa.png"), args.thr_b)

    write_readme(out_dir, args, len(files))
    print(f"[INFO] Processed files: {len(files)}")
    print(f"[INFO] Saved: {overall_csv}")
    print(f"[INFO] Saved: {by_bin_csv}")
    print(f"[INFO] Saved figures in: {fig_dir}")


if __name__ == "__main__":
    main()
