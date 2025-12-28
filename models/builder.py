# models/builder.py
from .hitchnet import HitchNet
from .bev_resnet import BEVResNet
from .imu_gru import IMUGRU
from .bev_imu_fusion import BEVIMUFusion


def build_hitchnet(mcfg):
    tcfg = mcfg["temporal"]
    scfg = mcfg["spatial"]
    fcfg = mcfg["fusion"]

    model = HitchNet(
        # Temporal
        micro_input_dim=tcfg.get("micro_input_dim", 5),
        micro_hidden_dim=tcfg.get("micro_hidden_dim", 64),
        micro_layers=tcfg.get("micro_layers", 1),
        bidirectional_micro=tcfg.get("bidirectional_micro", False),

        macro_hidden_dim=tcfg.get("macro_hidden_dim", 128),
        macro_layers=tcfg.get("macro_layers", 2),

        # Spatial
        gat_hidden_dim=scfg.get("gat_hidden_dim", 128),
        gat_layers=scfg.get("gat_layers", 3),
        gat_heads=scfg.get("gat_heads", 4),
        gat_k=scfg.get("gat_k", 16),

        # Fusion
        fusion_dim=fcfg.get("fusion_dim", 256),
        fusion_heads=fcfg.get("fusion_heads", 4),
        dropout=fcfg.get("dropout", 0.1),
    )
    return model


def build_model(mcfg):
    """
    Unified builder:
      model:
        name: hitchnet | bev_resnet | imu_gru | bev_imu_fusion
    """
    name = mcfg.get("name", "hitchnet")

    if name == "hitchnet":
        return build_hitchnet(mcfg)

    if name == "bev_resnet":
        return BEVResNet(
            in_channels=mcfg.get("bev_channels", 3),
            backbone=mcfg.get("resnet", 18),
            feat_dim=mcfg.get("feat_dim", 256),
            head_hidden=mcfg.get("head_hidden", 128),
            dropout=mcfg.get("dropout", 0.1),
        )

    if name == "imu_gru":
        return IMUGRU(
            imu_dim=mcfg.get("imu_dim", 3),
            hidden_dim=mcfg.get("hidden_dim", 128),
            num_layers=mcfg.get("num_layers", 1),
            dropout=mcfg.get("dropout", 0.0),
            head_hidden=mcfg.get("head_hidden", 128),
        )

    if name == "bev_imu_fusion":
        return BEVIMUFusion(
            bev_in_channels=mcfg.get("bev_channels", 3),
            bev_backbone=mcfg.get("resnet", 18),
            bev_feat_dim=mcfg.get("bev_feat_dim", 256),
            imu_dim=mcfg.get("imu_dim", 3),
            imu_hidden_dim=mcfg.get("imu_hidden_dim", 128),
            imu_layers=mcfg.get("imu_layers", 1),
            fusion=mcfg.get("fusion", "late"),  # 'late' or 'gated'
            head_hidden=mcfg.get("head_hidden", 128),
            dropout=mcfg.get("dropout", 0.1),
        )

    raise ValueError(f"Unknown model name: {name}")
