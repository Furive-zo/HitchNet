import torch
import torch.nn as nn
from torch.autograd import Function

from .bev_resnet import ResNetBackbone


class _GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None


def grad_reverse(x, lambd=1.0):
    return _GradReverse.apply(x, float(lambd))


class BEVResNetDANN(nn.Module):
    """
    BEV backbone + angle regressor + domain classifier (DANN).
    - forward(batch): returns (B,2) [cos, sin] for eval/test compatibility
    - forward_with_domain(batch, grl_lambda): returns angle and domain logits
    """

    def __init__(
        self,
        in_channels=3,
        backbone=18,
        feat_dim=256,
        head_hidden=128,
        domain_hidden=128,
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

        self.domain_head = nn.Sequential(
            nn.Linear(feat_dim, domain_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(domain_hidden, 2),
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

    def forward_with_domain(self, batch, grl_lambda=1.0):
        z = self.encode(batch)
        angle = self._normalize_sincos(self.angle_head(z))
        dom = self.domain_head(grad_reverse(z, grl_lambda))
        return angle, dom

