# models/bev_resnet_regression_geom.py
import torch
import torch.nn as nn

from .bev_resnet import ResNetBackbone


class BEVResNetRegressionGeom(nn.Module):
    """
    BEV-only regression with centroid-angle hint.
      - BEV encoder
      - centroid angle (sin/cos) concatenated as feature
    output: (B,2) [cos, sin]
    """
    def __init__(
        self,
        in_channels=3,
        backbone=18,
        feat_dim=256,
        head_hidden=128,
        dropout=0.1,
    ):
        super().__init__()
        self.encoder = ResNetBackbone(in_channels=in_channels, depth=backbone, feat_dim=feat_dim)
        self.head = nn.Sequential(
            nn.Linear(feat_dim + 2, head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 2),
        )

    @staticmethod
    def _normalize_sincos(x, eps=1e-6):
        n = torch.norm(x, dim=-1, keepdim=True).clamp_min(eps)
        return x / n

    @staticmethod
    def _centroid_angle(batch):
        centroid_xy = batch.get("centroid_xy")
        if centroid_xy is not None:
            return torch.atan2(-centroid_xy[:, 1], -centroid_xy[:, 0])

        pcd = batch["pcd"]
        mask = batch.get("pcd_mask")
        if mask is not None:
            m = mask.to(pcd.dtype)
            x = pcd[:, :, 0]
            y = pcd[:, :, 1]
            neg_inf = torch.full_like(x, -1e9)
            pos_inf = torch.full_like(x, 1e9)
            x_min = torch.where(m > 0, x, pos_inf).min(dim=1).values
            x_max = torch.where(m > 0, x, neg_inf).max(dim=1).values
            y_min = torch.where(m > 0, y, pos_inf).min(dim=1).values
            y_max = torch.where(m > 0, y, neg_inf).max(dim=1).values
            cx = (x_min + x_max) * 0.5
            cy = (y_min + y_max) * 0.5
        else:
            x = pcd[:, :, 0]
            y = pcd[:, :, 1]
            cx = (x.min(dim=1).values + x.max(dim=1).values) * 0.5
            cy = (y.min(dim=1).values + y.max(dim=1).values) * 0.5

        return torch.atan2(-cy, -cx)

    def forward(self, batch):
        bev = batch["bev"]
        z = self.encoder(bev)
        theta0 = self._centroid_angle(batch)
        c = torch.cos(theta0)
        s = torch.sin(theta0)
        z = torch.cat([z, c.unsqueeze(-1), s.unsqueeze(-1)], dim=-1)
        out = self.head(z)
        out = self._normalize_sincos(out)
        return out
