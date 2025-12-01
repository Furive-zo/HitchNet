# utils/loss.py

import torch
import torch.nn.functional as F

def hitch_loss(pred, gt, lambda_norm=0.01):
    """
    pred: (B,2) -> [cosθ, sinθ]
    gt:   (B,2)
    """

    # 1) MSE for cos/sin
    mse = F.mse_loss(pred, gt)

    # 2) Norm regularization (encourage unit-length vector)
    pred_norm = torch.linalg.norm(pred, dim=1)
    norm_loss = ((pred_norm - 1.0)**2).mean()

    return mse + lambda_norm * norm_loss
