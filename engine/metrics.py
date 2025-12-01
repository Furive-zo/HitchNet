# engine/metrics.py

import numpy as np


def compute_angle_metrics(pred, gt):
    """
    pred: (N,2)  [cos,sin]
    gt  : (N,2)  [cos,sin]
    """

    # Convert cos/sin → angle (rad)
    pred_theta = np.arctan2(pred[:, 1], pred[:, 0])
    gt_theta = np.arctan2(gt[:, 1], gt[:, 0])

    diff = pred_theta - gt_theta

    # wrap (-pi,pi)
    diff = (diff + np.pi) % (2*np.pi) - np.pi

    rmse = np.sqrt(np.mean(diff**2))
    mae = np.mean(np.abs(diff))
    max_err = np.max(np.abs(diff))
    min_err = np.min(np.abs(diff))

    return {
        "rmse_rad": rmse,
        "rmse_deg": rmse * 180/np.pi,
        "mae_deg": mae * 180/np.pi,
        "max_err_deg": max_err * 180/np.pi,
        "min_err_deg": min_err * 180/np.pi
    }
