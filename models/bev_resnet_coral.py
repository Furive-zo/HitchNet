import torch
import torch.nn as nn

from .bev_resnet import ResNetBackbone


class BEVResNetCORAL(nn.Module):
    """
    BEV backbone + angle regressor for Deep CORAL training.
    - forward(batch): (B,2) normalized [cos, sin]
    - forward_with_feat(batch): ((B,2), (B,D)) for CORAL loss
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
        self.angle_head = nn.Sequential(
            nn.Linear(feat_dim, head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 2),
        )

    @staticmethod
    def _normalize_sincos(x, eps=1e-6):
        n = torch.norm(x, dim=-1, keepdim=True).clamp_min(eps)
        return x / n

    def encode(self, batch):
        return self.encoder(batch["bev"])

    def forward(self, batch):
        z = self.encode(batch)
        out = self.angle_head(z)
        return self._normalize_sincos(out)

    def forward_with_feat(self, batch):
        z = self.encode(batch)
        out = self._normalize_sincos(self.angle_head(z))
        return out, z
