import torch
import torch.nn.functional as F
import numpy as np

def wrap_rad_torch(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def hitch_loss(
    pred_sc: torch.Tensor,          # [B,2] = (cos_hat, sin_hat)
    gt_sc: torch.Tensor,            # [B,2] = (cos, sin)
    weight_factor: float = 3.0,
    alpha: float = 0.7,
    beta: float = 0.3,
    clamp_w: float = 8.0,
    huber_delta_deg: float = 1.0
):
    """
    pred_sc: (B,2) [cos_hat, sin_hat]
    gt_sc:   (B,2) [cos, sin]
    """

    # --------------------------------------
    # 1) angle difference Δθ(rad)
    # --------------------------------------
    pred_cos, pred_sin = pred_sc[:, 0], pred_sc[:, 1]
    gt_cos,   gt_sin   = gt_sc[:, 0], gt_sc[:, 1]

    theta_p = torch.atan2(pred_sin, pred_cos)
    theta_g = torch.atan2(gt_sin, gt_cos)

    delta_rad = wrap_rad_torch(theta_p - theta_g)  # (B)

    # --------------------------------------
    # 2) difficulty-aware weight (based on |θ_gt|)
    # --------------------------------------
    w = 1.0 + weight_factor * (torch.abs(theta_g) / np.pi)
    if clamp_w is not None:
        w = torch.clamp(w, max=clamp_w)

    # --------------------------------------
    # 3) Huber Loss (smooth L1) on Δθ(rad)
    # --------------------------------------
    d = huber_delta_deg * np.pi / 180.0
    d_t = torch.tensor(d, device=delta_rad.device)

    abs_delta = torch.abs(delta_rad)

    quad = torch.minimum(abs_delta, d_t)
    lin = torch.clamp(abs_delta - d_t, min=0.0)
    huber_elem = 0.5 * quad**2 + d_t * lin

    # weighted Huber
    huber_w = torch.mean(w * huber_elem)

    # --------------------------------------
    # 4) cosine similarity auxiliary term
    # --------------------------------------
    pred_unit = pred_sc / (pred_sc.norm(dim=-1, keepdim=True) + 1e-8)
    cos_sim = (pred_unit * gt_sc).sum(dim=-1)
    cos_loss = (1.0 - cos_sim).mean()

    # --------------------------------------
    # 5) total loss
    # --------------------------------------
    loss = alpha * huber_w + beta * cos_loss

    return loss
