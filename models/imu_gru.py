# models/imu_gru.py
import torch
import torch.nn as nn


class IMUGRU(nn.Module):
    """
    IMU-only baseline:
    batch['imu'] : (B,T,M,3)  (your current dataset style)
      - we pool over M (micro_seq) -> (B,T,3)
    output: (B,2) [sin, cos]
    """
    def __init__(self, imu_dim=3, hidden_dim=128, num_layers=1, dropout=0.0, head_hidden=128):
        super().__init__()
        self.gru = nn.GRU(
            input_size=imu_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, head_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(head_hidden, 2),
        )

    @staticmethod
    def _normalize_sincos(x, eps=1e-6):
        n = torch.norm(x, dim=-1, keepdim=True).clamp_min(eps)
        return x / n

    def forward(self, batch):
        imu = batch["imu"]  # (B,T,M,3)
        if imu.dim() == 4:
            imu = imu.mean(dim=2)  # (B,T,3)

        out, h = self.gru(imu)      # h: (L,B,H)
        z = h[-1]                   # (B,H)
        y = self.head(z)            # (B,2)
        return self._normalize_sincos(y)
