#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(description="Plot planning-proxy figures from planning_stability_summary CSV.")
    p.add_argument("--summary_csv", type=str, required=True)
    p.add_argument("--thr", type=int, default=35)
    p.add_argument("--out_dir", type=str, default="results/planning_stability_custom")
    return p.parse_args()


def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def fget(row, key, default=np.nan):
    v = row.get(key, "")
    if v in ("", "nan", "NaN", None):
        return float(default)
    return float(v)


def main():
    args = parse_args()
    rows = read_rows(args.summary_csv)
    thr = int(args.thr)
    os.makedirs(args.out_dir, exist_ok=True)

    # Auto-fallback threshold if requested one is not in CSV.
    header = rows[0].keys() if rows else []
    avail = []
    for k in header:
        if k.startswith("miss_rate_") and k.endswith("_mean"):
            try:
                avail.append(int(k[len("miss_rate_"):-len("_mean")]))
            except ValueError:
                pass
    if avail and thr not in avail:
        # choose closest available threshold
        thr_old = thr
        thr = min(avail, key=lambda x: abs(x - thr_old))
        print(f"[WARN] Requested thr={thr_old} not found in CSV. Using thr={thr}.")

    groups = ["low_curvature", "high_curvature", "elevated"]
    glabel = {
        "low_curvature": "Low curvature",
        "high_curvature": "High curvature",
        "elevated": "Elevated",
    }

    methods = []
    for r in rows:
        if r["group"] == "overall" and r["method"] not in methods:
            methods.append(r["method"])
    # Keep GT first if present, then others.
    methods = sorted(methods, key=lambda m: (0 if m == "GT" else 1, m))
    cmap = plt.get_cmap("tab10")
    color = {m: cmap(i % 10) for i, m in enumerate(methods)}

    # Figure 1: Miss / False alarm by scenario
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0), sharex=True)
    x = np.arange(len(groups))
    bw = 0.78 / max(len(methods), 1)
    for mi, m in enumerate(methods):
        off = -0.39 + (mi + 0.5) * bw
        miss_vals, fa_vals = [], []
        for g in groups:
            row = next((r for r in rows if r["method"] == m and r["group"] == g), None)
            miss_vals.append(fget(row, f"miss_rate_{thr}_mean", np.nan) * 100.0 if row else np.nan)
            fa_vals.append(fget(row, f"false_alarm_rate_{thr}_mean", np.nan) * 100.0 if row else np.nan)
        axes[0].bar(x + off, miss_vals, width=bw * 0.9, color=color[m], edgecolor="#222", linewidth=0.5, label=m)
        axes[1].bar(x + off, fa_vals, width=bw * 0.9, color=color[m], edgecolor="#222", linewidth=0.5, label=m)

    axes[0].set_ylabel(f"Miss Rate @ {thr}° (%)")
    axes[1].set_ylabel(f"False Alarm Rate @ {thr}° (%)")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([glabel[g] for g in groups])
        ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].legend(frameon=False, fontsize=9, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, f"planning_miss_fa_by_scenario_{thr}.png"), dpi=220)
    plt.close(fig)

    # Figure 2: overall lead-detect vs lead-time scatter
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    for m in methods:
        row = next((r for r in rows if r["method"] == m and r["group"] == "overall"), None)
        if row is None:
            continue
        xval = fget(row, f"lead_detect_rate_{thr}_mean", np.nan) * 100.0
        yval = fget(row, f"lead_time_to_risk_{thr}_mean", np.nan)
        if not np.isfinite(xval) or not np.isfinite(yval):
            continue
        ax.scatter([xval], [yval], s=70, color=color[m], edgecolors="#222", linewidths=0.6)
        ax.text(xval + 0.8, yval, m, fontsize=9, va="center")
    ax.set_xlabel(f"Lead Detection Rate @ {thr}° (%)")
    ax.set_ylabel(f"Lead Time to Risk @ {thr}° (frames)")
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, f"planning_lead_scatter_{thr}.png"), dpi=220)
    plt.close(fig)

    print(f"[INFO] Saved: {os.path.join(args.out_dir, f'planning_miss_fa_by_scenario_{thr}.png')}")
    print(f"[INFO] Saved: {os.path.join(args.out_dir, f'planning_lead_scatter_{thr}.png')}")


if __name__ == "__main__":
    main()
