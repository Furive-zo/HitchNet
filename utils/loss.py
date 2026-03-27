import torch
import torch.nn.functional as F
import numpy as np

def wrap_rad_torch(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def hitch_loss(
    pred_sc: torch.Tensor,          # [B,2] = (cos_hat, sin_hat)
    gt_sc: torch.Tensor,            # [B,2] = (cos, sin)
    weight_factor: float = 3.0,
    alpha: float = 1.0,
    beta: float = 0.0,
    clamp_w: float = 8.0,
    huber_delta_deg: float = 0.5,
    bins: torch.Tensor | None = None,
    bin_counts: torch.Tensor | None = None,
    bin_alpha: float = 0.0,
    bin_clamp: float = 8.0,
    pcd: torch.Tensor | None = None,        # [B,N,3]
    pcd_mask: torch.Tensor | None = None,   # [B,N]
    line_loss_w: float = 0.0,
    bbox_loss_w: float = 0.0,
    bbox_len: float | None = None,
    bbox_wid: float | None = None,
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

    # --------------------------------------
    # 6) optional line distance consistency (signed)
    # --------------------------------------
    if line_loss_w > 0.0 and pcd is not None:
        pts_xy = pcd[..., :2]  # (B,N,2)
        if pcd_mask is not None:
            mask = pcd_mask.unsqueeze(-1)  # (B,N,1)
            denom = mask.sum(dim=1).clamp_min(1.0)
            centroid = (pts_xy * mask).sum(dim=1) / denom  # (B,2)
        else:
            centroid = pts_xy.mean(dim=1)

        v_pred = pred_sc / (pred_sc.norm(dim=-1, keepdim=True) + 1e-8)
        v_gt = gt_sc / (gt_sc.norm(dim=-1, keepdim=True) + 1e-8)
        # signed distance to line through origin with direction v
        d_pred = v_pred[:, 0] * centroid[:, 1] - v_pred[:, 1] * centroid[:, 0]
        d_gt = v_gt[:, 0] * centroid[:, 1] - v_gt[:, 1] * centroid[:, 0]
        line_loss = torch.mean(torch.abs(d_pred - d_gt))
        loss = loss + line_loss_w * line_loss

    # --------------------------------------
    # 7) optional bbox loss (outside distance sum)
    # --------------------------------------
    if bbox_loss_w > 0.0 and pcd is not None and bbox_len is not None and bbox_wid is not None:
        pts_xy = pcd[..., :2]  # (B,N,2)
        if pcd_mask is not None:
            mask = pcd_mask.unsqueeze(-1).float()  # (B,N,1)
        else:
            mask = torch.ones_like(pts_xy[..., :1])

        # rotate points into box frame (aligned with pred angle)
        pred_unit = pred_sc / (pred_sc.norm(dim=-1, keepdim=True) + 1e-8)
        c, s = pred_unit[:, 0], pred_unit[:, 1]
        rot = torch.stack(
            [torch.stack([c, s], dim=-1), torch.stack([-s, c], dim=-1)],
            dim=-2,
        )  # (B,2,2) rotation by -theta
        pts_local = torch.einsum("bij,bnj->bni", rot, pts_xy)  # (B,N,2)

        x = pts_local[..., 0]
        y = pts_local[..., 1]
        x_max = 0.0
        x_min = -bbox_len
        y_max = bbox_wid / 2.0

        dx = torch.clamp(x - x_max, min=0.0) + torch.clamp(x_min - x, min=0.0)
        dy = torch.clamp(torch.abs(y) - y_max, min=0.0)
        dist = (dx + dy) * mask.squeeze(-1)

        denom = mask.squeeze(-1).sum(dim=1).clamp_min(1.0)
        bbox_loss = (dist.sum(dim=1) / denom).mean()
        loss = loss + bbox_loss_w * bbox_loss

    return loss
