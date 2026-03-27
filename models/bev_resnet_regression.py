# models/bev_resnet_regression.py
import torch
import torch.nn as nn

from .bev_resnet import ResNetBackbone


class BEVResNetRegression(nn.Module):
    """
    BEV-only regression (no centroid/geometry hint).
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
            nn.Linear(feat_dim, head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 2),
        )

    @staticmethod
    def _normalize_sincos(x, eps=1e-6):
        n = torch.norm(x, dim=-1, keepdim=True).clamp_min(eps)
        return x / n

    def forward(self, batch):
        bev = batch["bev"]
        z = self.encoder(bev)
        out = self.head(z)
        out = self._normalize_sincos(out)
        return out
