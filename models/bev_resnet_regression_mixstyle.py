import torch
import torch.nn as nn

from .bev_resnet import BasicBlock
from .mixstyle import MixStyle


class ResNetBackboneMixStyle(nn.Module):
    """
    ResNet backbone with optional MixStyle insertion at intermediate stages.
    """

    def __init__(
        self,
        in_channels=3,
        depth=18,
        feat_dim=256,
        mixstyle_p=0.5,
        mixstyle_alpha=0.1,
        mixstyle_layer2=True,
        mixstyle_layer3=True,
    ):
        super().__init__()
        assert depth in (18, 34), "depth must be 18 or 34"

        layers = {18: [2, 2, 2, 2], 34: [3, 4, 6, 3]}[depth]
        self.in_planes = 64

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = self._make_layer(64, layers[0], stride=1)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)

        self.mixstyle = MixStyle(p=mixstyle_p, alpha=mixstyle_alpha)
        self.use_ms_l2 = bool(mixstyle_layer2)
        self.use_ms_l3 = bool(mixstyle_layer3)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(512, feat_dim) if feat_dim != 512 else nn.Identity()

    def _make_layer(self, planes, blocks, stride):
        layers = [BasicBlock(self.in_planes, planes, stride=stride)]
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_planes, planes, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        if self.use_ms_l2:
            x = self.mixstyle(x)
        x = self.layer3(x)
        if self.use_ms_l3:
            x = self.mixstyle(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)
        x = self.proj(x)
        return x


class BEVResNetRegressionMixStyle(nn.Module):
    """
    BEV regression with MixStyle regularization for DG.
    output: (B,2) [cos, sin]
    """

    def __init__(
        self,
        in_channels=3,
        backbone=18,
        feat_dim=256,
        head_hidden=128,
        dropout=0.1,
        mixstyle_p=0.5,
        mixstyle_alpha=0.1,
        mixstyle_layer2=True,
        mixstyle_layer3=True,
    ):
        super().__init__()
        self.encoder = ResNetBackboneMixStyle(
            in_channels=in_channels,
            depth=backbone,
            feat_dim=feat_dim,
            mixstyle_p=mixstyle_p,
            mixstyle_alpha=mixstyle_alpha,
            mixstyle_layer2=mixstyle_layer2,
            mixstyle_layer3=mixstyle_layer3,
        )
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
        return self._normalize_sincos(out)
