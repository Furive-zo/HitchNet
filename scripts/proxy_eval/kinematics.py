#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Kinematic utilities for tractor/trailer proxy reconstruction."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def validate_time_monotonic(t: np.ndarray) -> None:
    """Raise ValueError if timestamps are not strictly increasing."""
    dt = np.diff(t)
    if np.any(~np.isfinite(dt)) or np.any(dt <= 0.0):
        raise ValueError("Timestamp must be strictly increasing and finite.")


def integrate_pose_dead_reckoning(t: np.ndarray, v: np.ndarray, yaw_rate: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Dead-reckon tractor pose from speed and yaw-rate.

    Uses forward Euler:
      phi[k+1] = phi[k] + yaw_rate[k] * dt
      x[k+1]   = x[k] + v[k] * cos(phi[k]) * dt
      y[k+1]   = y[k] + v[k] * sin(phi[k]) * dt
    """
    validate_time_monotonic(t)
    n = t.shape[0]
    x = np.zeros(n, dtype=np.float64)
    y = np.zeros(n, dtype=np.float64)
    phi = np.zeros(n, dtype=np.float64)
    dt = np.diff(t)
    phi[1:] = np.cumsum(yaw_rate[:-1] * dt)
    x[1:] = np.cumsum(v[:-1] * np.cos(phi[:-1]) * dt)
    y[1:] = np.cumsum(v[:-1] * np.sin(phi[:-1]) * dt)
    return x, y, phi


def use_pose_from_log(log: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return x,y,yaw directly from log pose columns."""
    return log["x"].astype(np.float64), log["y"].astype(np.float64), log["yaw"].astype(np.float64)


def trailer_axle_from_gamma(
    x: np.ndarray,
    y: np.ndarray,
    phi: np.ndarray,
    gamma: np.ndarray,
    L2: float,
    eps: float,
    gamma_sign: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute hitch and trailer axle positions from tractor pose and hitch angle.

    theta = phi - gamma_sign * gamma
    hitch: (xh, yh) = (x - eps*cos(phi), y - eps*sin(phi))
    trailer axle: (xtr, ytr) = (xh - L2*cos(theta), yh - L2*sin(theta))
    """
    if gamma_sign not in (-1, 1):
        raise ValueError("--gamma_sign must be +1 or -1")

    xh = x - float(eps) * np.cos(phi)
    yh = y - float(eps) * np.sin(phi)
    theta = phi - float(gamma_sign) * gamma
    xtr = xh - float(L2) * np.cos(theta)
    ytr = yh - float(L2) * np.sin(theta)
    return xtr, ytr, xh, yh


def compute_proxy_timeseries(
    log: Dict[str, np.ndarray],
    L2: float,
    eps: float,
    gamma_sign: int,
    v_min: float,
) -> Dict[str, np.ndarray]:
    """
    Build full proxy timeseries: GT/pred trailer axle, ey, kappa, valid mask.
    """
    t = log["t"].astype(np.float64)
    v = log["v"].astype(np.float64)
    yaw_rate = log["yaw_rate"].astype(np.float64)
    gamma_gt = log["gamma_gt"].astype(np.float64)
    gamma_pred = log["gamma_pred"].astype(np.float64)

    if {"x", "y", "yaw"}.issubset(log.keys()):
        x, y, phi = use_pose_from_log(log)
        validate_time_monotonic(t)
    else:
        x, y, phi = integrate_pose_dead_reckoning(t, v, yaw_rate)

    x_gt, y_gt, xh, yh = trailer_axle_from_gamma(x, y, phi, gamma_gt, L2=L2, eps=eps, gamma_sign=gamma_sign)
    x_pr, y_pr, _, _ = trailer_axle_from_gamma(x, y, phi, gamma_pred, L2=L2, eps=eps, gamma_sign=gamma_sign)

    dx = x_pr - x_gt
    dy = y_pr - y_gt
    # Lateral normal of tractor heading.
    nx = -np.sin(phi)
    ny = np.cos(phi)
    ey = nx * dx + ny * dy
    kappa = yaw_rate / np.maximum(np.abs(v), 1e-12)

    finite = (
        np.isfinite(t)
        & np.isfinite(v)
        & np.isfinite(yaw_rate)
        & np.isfinite(gamma_gt)
        & np.isfinite(gamma_pred)
        & np.isfinite(phi)
        & np.isfinite(x_gt)
        & np.isfinite(y_gt)
        & np.isfinite(x_pr)
        & np.isfinite(y_pr)
        & np.isfinite(ey)
        & np.isfinite(kappa)
    )
    valid = finite & (np.abs(v) >= float(v_min))

    return {
        "t": t,
        "x": x,
        "y": y,
        "phi": phi,
        "v": v,
        "yaw_rate": yaw_rate,
        "kappa": kappa,
        "gamma_gt": gamma_gt,
        "gamma_pred": gamma_pred,
        "xh": xh,
        "yh": yh,
        "xtr_gt": x_gt,
        "ytr_gt": y_gt,
        "xtr_pred": x_pr,
        "ytr_pred": y_pr,
        "ey": ey,
        "valid_mask": valid,
    }

