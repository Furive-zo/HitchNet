#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Plotting utilities for proxy evaluation outputs."""

from __future__ import annotations

import os
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"],
    "font.size": 12,
})

METHOD_COLORS = {
    "Naive_BEV": "#1F4E79",
    "Naive BEV": "#1F4E79",
    "CORAL-UDA": "#2E8B57",
    "CORAL-DG": "#7A5EA8",
    "MixStyle": "#7A5EA8",
    "Ours_(Alignment_only)": "#F4A261",
    "Ours (Alignment only)": "#F4A261",
    "Ours_(Full)": "#D62828",
    "Ours (Full)": "#D62828",
}

METHOD_MARKERS = {
    "Naive_BEV": "o",
    "Naive BEV": "o",
    "CORAL-UDA": "s",
    "CORAL-DG": "^",
    "MixStyle": "^",
    "Ours_(Alignment_only)": "D",
    "Ours (Alignment only)": "D",
    "Ours_(Full)": "*",
    "Ours (Full)": "*",
}

CHARCOAL = "#222222"


def _group_by_method(rows: List[Dict]) -> Dict[str, List[Dict]]:
    g: Dict[str, List[Dict]] = {}
    for r in rows:
        g.setdefault(r["method"], []).append(r)
    return g


def _is_ours_full(method: str) -> bool:
    return method in ("Ours_(Full)", "Ours (Full)")


def _method_rank(method: str) -> int:
    order = {
        "Naive_BEV": 0,
        "Naive BEV": 0,
        "CORAL-DG": 1,
        "CORAL-UDA": 2,
        "Ours_(Alignment_only)": 3,
        "Ours (Alignment only)": 3,
        "Ours_(Full)": 4,
        "Ours (Full)": 4,
        "MixStyle": 5,
    }
    return order.get(method, 999)


def _display_name(method: str) -> str:
    mapping = {
        "Naive_BEV": "Naive BEV",
        "Naive BEV": "Naive BEV",
        "CORAL-UDA": "CORAL-UDA",
        "CORAL-DG": "CORAL-DG",
        "MixStyle": "MixStyle",
        "Ours_(Alignment_only)": "Ours(alignment only)",
        "Ours (Alignment only)": "Ours(alignment only)",
        "Ours_(Full)": "Ours(Full)",
        "Ours (Full)": "Ours(Full)",
    }
    return mapping.get(method, method)


def plot_p95_vs_kappa(rows_by_bin: List[Dict], out_path: str) -> None:
    """Plot p95(|ey|) by |kappa| bin center with sample-count subplot."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    grouped = _group_by_method(rows_by_bin)

    fig, (ax_main, ax_n) = plt.subplots(
        2, 1, figsize=(7.4, 5.6), sharex=True, gridspec_kw={"height_ratios": [3.3, 1.0]}
    )

    kappa_risk_start = 0.006
    ax_main.axvspan(kappa_risk_start, 1e9, color="#d9d9d9", alpha=0.28, zorder=0)
    ax_n.axvspan(kappa_risk_start, 1e9, color="#d9d9d9", alpha=0.28, zorder=0)
    ax_main.axvline(kappa_risk_start, color="#7a7a7a", linestyle="--", linewidth=1.0, alpha=0.9, zorder=1)
    ax_n.axvline(kappa_risk_start, color="#7a7a7a", linestyle="--", linewidth=1.0, alpha=0.9, zorder=1)

    all_x_for_xlim = []

    for method in sorted(grouped.keys(), key=_method_rank):
        rows = grouped[method]
        rows = sorted(rows, key=lambda r: r["kappa_bin_center"])

        # keep only statistically valid bins
        rows = [
            r for r in rows
            if float(r["n"]) >= 50 and str(r["p95_abs_ey"]).lower() not in ("nan", "")
        ]
        if not rows:
            continue

        x = np.array([max(float(r["kappa_bin_center"]), 1e-4) for r in rows], dtype=float)
        y = np.array([float(r["p95_abs_ey"]) for r in rows], dtype=float)
        all_x_for_xlim.extend(x.tolist())

        if _is_ours_full(method):
            ax_main.plot(
                x, y, marker=METHOD_MARKERS.get(method, "o"), markersize=8.8, linewidth=3.0, linestyle="-",
                color=METHOD_COLORS.get(method, None), label=_display_name(method), zorder=4
            )
        else:
            ax_main.plot(
                x, y, marker=METHOD_MARKERS.get(method, "o"), markersize=6.5, linewidth=2.2, linestyle="-",
                color=METHOD_COLORS.get(method, None), label=_display_name(method), zorder=3
            )

    ax_main.set_ylabel(r"p95($|e_y|$) [m]")
    y_max = max([np.nanmax(np.array([r["p95_abs_ey"] for r in rows], dtype=float)) for rows in grouped.values()] + [0.27])
    ax_main.set_ylim(0.04, y_max * 1.02)
    ax_main.grid(True, alpha=0.25, linestyle="--", linewidth=0.8)
    ax_main.spines["top"].set_visible(False)
    ax_main.spines["right"].set_visible(False)
    ax_main.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=max(len(grouped), 1),
    )

    # n subplot
    bin_map: Dict[float, List[float]] = {}
    for rows in grouped.values():
        for r in rows:
            c = float(r["kappa_bin_center"])
            bin_map.setdefault(c, []).append(float(r["n"]))
    x_n = np.array([max(c, 1e-4) for c in sorted(bin_map.keys())], dtype=float)
    n_vals = np.array([np.median(bin_map[c]) for c in x_n], dtype=float)

    ax_n.plot(x_n, n_vals, color=CHARCOAL, marker="o", markersize=2.6, linewidth=1.4)
    ax_n.set_yscale("log")
    ax_main.set_xscale("log")
    ax_n.set_xscale("log")

    ax_main.set_xlim(1e-3, 1e-2)
    ax_n.set_xlim(1e-3, 1e-2)
    ax_n.set_ylabel("n")
    ax_n.set_xlabel(r"$|\kappa|$ (1/m)")
    ax_n.grid(True, alpha=0.25, linestyle="--", linewidth=0.8)
    ax_n.spines["top"].set_visible(False)
    ax_n.spines["right"].set_visible(False)

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
    fig.savefig(out_path, dpi=220)
    plt.close(fig)



def plot_exceedance_vs_kappa(rows_by_bin: List[Dict], out_path: str, threshold_m: float) -> None:
    """Plot exceedance rate by |kappa| bin center for each method."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    k = f"exceed_rate_{threshold_m:.1f}m"
    grouped = _group_by_method(rows_by_bin)
    plt.figure(figsize=(7.4, 4.4))
    for method in sorted(grouped.keys(), key=_method_rank):
        rows = grouped[method]
        rows = sorted(rows, key=lambda r: r["kappa_bin_center"])
        x = np.array([r["kappa_bin_center"] for r in rows], dtype=float)
        y = np.array([r[k] for r in rows], dtype=float) * 100.0
        is_full = _is_ours_full(method)
        plt.plot(
            x, y,
            marker=METHOD_MARKERS.get(method, "o"),
            markersize=(8.8 if is_full else 6.5),
            linewidth=(3.0 if is_full else 2.2),
            linestyle="-",
            color=METHOD_COLORS.get(method, None), label=_display_name(method)
        )
    plt.xlabel(r"$|\kappa|$ (1/m)")
    plt.ylabel(f"Exceedance Rate (|ey| > {threshold_m:.1f}m) [%]")
    plt.grid(True, alpha=0.25, linestyle="--", linewidth=0.8)
    ax = plt.gca()
    # Start y-axis at 0% as requested.
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(-0.2, ymax)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()
