# models/bev_resnet.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)

        self.downsample = None
        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = F.relu(out, inplace=True)
        return out


class ResNetBackbone(nn.Module):
    """
    Minimal ResNet-18/34 backbone for BEV.
    Returns a global feature vector (B, feat_dim).
    """
    def __init__(self, in_channels=3, depth=18, feat_dim=256):
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

        self.layer1 = self._make_layer(64,  layers[0], stride=1)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)

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
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)  # (B,512)
        x = self.proj(x)             # (B,feat_dim)
        return x


class BEVResNet(nn.Module):
    """
    LiDAR-only baseline:
    batch['bev'] : (B,C,H,W)
    output       : (B,2)  where [sin, cos]
    """
    def __init__(self, in_channels=3, backbone=18, feat_dim=256, head_hidden=128, dropout=0.1):
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
        # keep output close to unit circle (optional but stabilizes)
        n = torch.norm(x, dim=-1, keepdim=True).clamp_min(eps)
        return x / n

    def forward(self, batch):
        assert "bev" in batch, batch.keys()

        bev = batch["bev"]  # (B,C,H,W)
        z = self.encoder(bev)
        out = self.head(z)  # (B,2) raw
        out = self._normalize_sincos(out)
        return out
