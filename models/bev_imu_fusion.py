# models/bev_imu_fusion.py
import torch
import torch.nn as nn
from .bev_resnet import ResNetBackbone
from .imu_gru import IMUGRU


class BEVIMUFusion(nn.Module):
    """
    Fusion model:
      - BEV encoder (ResNet)
      - IMU encoder (GRU)
      - fusion: 'late' or 'gated'
    output: (B,2) [sin, cos]
    """
    def __init__(
        self,
        bev_in_channels=3,
        bev_backbone=18,
        bev_feat_dim=256,
        imu_dim=3,
        imu_hidden_dim=128,
        imu_layers=1,
        fusion="late",
        head_hidden=128,
        dropout=0.1,
    ):
        super().__init__()
        assert fusion in ("late", "gated")
        self.fusion = fusion

        self.bev_enc = ResNetBackbone(in_channels=bev_in_channels, depth=bev_backbone, feat_dim=bev_feat_dim)
        self.imu_enc = IMUGRU(imu_dim=imu_dim, hidden_dim=imu_hidden_dim, num_layers=imu_layers, head_hidden=imu_hidden_dim)

        # IMUGRU returns (B,2) if used directly; but we want features.
        # We'll replace IMUGRU head with identity and expose hidden feature through a lightweight wrapper:
        self.imu_gru = self.imu_enc.gru
        self.imu_feat = nn.Sequential(
            nn.Linear(imu_hidden_dim, imu_hidden_dim),
            nn.ReLU(inplace=True),
        )

        if fusion == "late":
            in_dim = bev_feat_dim + imu_hidden_dim
            self.head = nn.Sequential(
                nn.Linear(in_dim, head_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(head_hidden, 2),
            )
        else:
            # gated: gate from imu feature to scale BEV feature
            self.gate = nn.Sequential(
                nn.Linear(imu_hidden_dim, bev_feat_dim),
                nn.Sigmoid(),
            )
            self.head = nn.Sequential(
                nn.Linear(bev_feat_dim, head_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(head_hidden, 2),
            )

    @staticmethod
    def _normalize_sincos(x, eps=1e-6):
        n = torch.norm(x, dim=-1, keepdim=True).clamp_min(eps)
        return x / n

    def _encode_imu(self, imu):
        # imu: (B,T,M,3) or (B,T,3)
        if imu.dim() == 4:
            imu = imu.mean(dim=2)  # (B,T,3)
        _, h = self.imu_gru(imu)
        z = h[-1]                 # (B,H)
        z = self.imu_feat(z)      # (B,H)
        return z

    def forward(self, batch):
        bev = batch["bev"]        # (B,C,H,W)
        imu = batch["imu"]        # (B,T,M,3) or (B,T,3)

        z_bev = self.bev_enc(bev)        # (B,Db)
        z_imu = self._encode_imu(imu)    # (B,Di)

        if self.fusion == "late":
            z = torch.cat([z_bev, z_imu], dim=-1)
            y = self.head(z)
        else:
            g = self.gate(z_imu)         # (B,Db) in [0,1]
            z = z_bev * (1.0 + g)
            y = self.head(z)

        return self._normalize_sincos(y)
