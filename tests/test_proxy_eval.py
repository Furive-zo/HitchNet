#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from scripts.proxy_eval.kinematics import compute_proxy_timeseries


def make_straight_log(n=200, dt=0.1, v=5.0, gamma_gt=0.1, gamma_pred=0.1):
    t = np.arange(n, dtype=np.float64) * dt
    return {
        "t": t,
        "v": np.full(n, v, dtype=np.float64),
        "yaw_rate": np.zeros(n, dtype=np.float64),
        "gamma_gt": np.full(n, gamma_gt, dtype=np.float64),
        "gamma_pred": np.full(n, gamma_pred, dtype=np.float64),
    }


def test_zero_error_when_gamma_equal():
    log = make_straight_log(gamma_gt=0.12, gamma_pred=0.12)
    out = compute_proxy_timeseries(log, L2=6.2, eps=0.0, gamma_sign=1, v_min=0.5)
    ey = out["ey"][out["valid_mask"]]
    assert ey.size > 0
    assert np.max(np.abs(ey)) < 1e-8


def test_constant_bias_matches_L2_sin_bias_scale():
    bias = 0.05
    L2 = 6.2
    log = make_straight_log(gamma_gt=0.10, gamma_pred=0.10 + bias)
    out = compute_proxy_timeseries(log, L2=L2, eps=0.0, gamma_sign=1, v_min=0.5)
    ey = out["ey"][out["valid_mask"]]
    mean_abs = float(np.mean(np.abs(ey)))
    expected = abs(L2 * np.sin(bias))
    # Allow small numerical tolerance and sign convention effects.
    assert abs(mean_abs - expected) < 1e-3

