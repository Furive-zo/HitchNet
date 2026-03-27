#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Metrics for curvature-conditioned trailer deviation proxy."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import numpy as np


def overall_stats(ey: np.ndarray, thr_a: float = 0.2, thr_b: float = 0.3) -> Dict[str, float]:
    """Compute overall error statistics on ey [m]."""
    if ey.size == 0:
        return {
            "n": 0,
            "mean_abs_ey": np.nan,
            "rmse_ey": np.nan,
            "p95_abs_ey": np.nan,
            "p99_abs_ey": np.nan,
            f"exceed_rate_{thr_a:.1f}m": np.nan,
            f"exceed_rate_{thr_b:.1f}m": np.nan,
        }
    a = np.abs(ey)
    return {
        "n": int(ey.size),
        "mean_abs_ey": float(np.mean(a)),
        "rmse_ey": float(np.sqrt(np.mean(ey ** 2))),
        "p95_abs_ey": float(np.percentile(a, 95)),
        "p99_abs_ey": float(np.percentile(a, 99)),
        f"exceed_rate_{thr_a:.1f}m": float(np.mean(a > thr_a)),
        f"exceed_rate_{thr_b:.1f}m": float(np.mean(a > thr_b)),
    }


def curvature_binned_stats(
    ey: np.ndarray,
    kappa: np.ndarray,
    bin_edges: Sequence[float],
    min_n_per_bin: int = 50,
    thr_a: float = 0.2,
    thr_b: float = 0.3,
) -> List[Dict[str, float]]:
    """
    Compute |kappa|-conditioned bin stats for ey.
    """
    abs_k = np.abs(kappa)
    abs_e = np.abs(ey)
    out: List[Dict[str, float]] = []
    edges = np.asarray(bin_edges, dtype=np.float64)

    for i in range(len(edges) - 1):
        lo, hi = float(edges[i]), float(edges[i + 1])
        if i < len(edges) - 2:
            m = (abs_k >= lo) & (abs_k < hi)
        else:
            m = (abs_k >= lo) & (abs_k <= hi)
        n = int(np.sum(m))
        row = {
            "kappa_bin_lo": lo,
            "kappa_bin_hi": hi,
            "kappa_bin_center": 0.5 * (lo + hi),
            "n": n,
        }
        if n < int(min_n_per_bin):
            row["p95_abs_ey"] = np.nan
            row[f"exceed_rate_{thr_a:.1f}m"] = np.nan
            row[f"exceed_rate_{thr_b:.1f}m"] = np.nan
        else:
            e = abs_e[m]
            row["p95_abs_ey"] = float(np.percentile(e, 95))
            row[f"exceed_rate_{thr_a:.1f}m"] = float(np.mean(e > thr_a))
            row[f"exceed_rate_{thr_b:.1f}m"] = float(np.mean(e > thr_b))
        out.append(row)
    return out

