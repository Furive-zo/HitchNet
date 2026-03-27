# models/builder.py
# from .hitchnet import HitchNet
# from .bev_resnet import BEVResNet
# from .bev_resnet_residual import BEVResNetResidual
from .bev_resnet_regression import BEVResNetRegression
from .bev_resnet_regression_mixstyle import BEVResNetRegressionMixStyle
from .bev_resnet_regression_geom import BEVResNetRegressionGeom
from .bev_resnet_dann import BEVResNetDANN
from .bev_resnet_coral import BEVResNetCORAL
# from .pointpillars_bev_regression import PointPillarsBEVRegression
# from .imu_gru import IMUGRU
# from .bev_imu_fusion import BEVIMUFusion
# from .bev_imu_fusion_residual import BEVIMUFusionResidual
# from .bev_imu_fusion_regression import BEVIMUFusionRegression
# from .hitch_query_transformer import HitchQueryTransformer
from .rule_based import RuleBasedCentroid, RuleBasedPCA, RuleBasedOLS, RuleBasedMLE


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
        name: hitchnet | bev_resnet | bev_resnet_residual | imu_gru | bev_imu_fusion | bev_imu_fusion_residual | bev_resnet_regression_geom | pointpillars_bev_regression | pointnet_imu_fusion | rule_based | rule_based_pca | rule_based_ols | rule_based_mle | bev_resnet_coral | bev_resnet_regression_mixstyle
    """
    name = mcfg.get("name", "hitchnet")

    # if name == "hitchnet":
    #     return build_hitchnet(mcfg)

    # if name == "bev_resnet":
    #     return BEVResNet(
    #         in_channels=mcfg.get("bev_channels", 3),
    #         backbone=mcfg.get("resnet", 18),
    #         feat_dim=mcfg.get("feat_dim", 256),
    #         head_hidden=mcfg.get("head_hidden", 128),
    #         dropout=mcfg.get("dropout", 0.1),
    #     )

    # if name == "bev_resnet_residual":
    #     return BEVResNetResidual(
    #         bev_in_channels=mcfg.get("bev_channels", 3),
    #         bev_backbone=mcfg.get("resnet", 18),
    #         bev_feat_dim=mcfg.get("bev_feat_dim", 256),
    #         head_hidden=mcfg.get("head_hidden", 128),
    #         dropout=mcfg.get("dropout", 0.1),
    #         delta_max_deg=mcfg.get("delta_max_deg", 30.0),
    #     )
    if name == "bev_resnet_regression":
        return BEVResNetRegression(
            in_channels=mcfg.get("bev_channels", 3),
            backbone=mcfg.get("resnet", 18),
            feat_dim=mcfg.get("feat_dim", 256),
            head_hidden=mcfg.get("head_hidden", 128),
            dropout=mcfg.get("dropout", 0.1),
        )
    if name == "bev_resnet_regression_mixstyle":
        return BEVResNetRegressionMixStyle(
            in_channels=mcfg.get("bev_channels", 3),
            backbone=mcfg.get("resnet", 18),
            feat_dim=mcfg.get("feat_dim", 256),
            head_hidden=mcfg.get("head_hidden", 128),
            dropout=mcfg.get("dropout", 0.1),
            mixstyle_p=mcfg.get("mixstyle_p", 0.5),
            mixstyle_alpha=mcfg.get("mixstyle_alpha", 0.1),
            mixstyle_layer2=mcfg.get("mixstyle_layer2", True),
            mixstyle_layer3=mcfg.get("mixstyle_layer3", True),
        )
    if name == "bev_resnet_regression_geom":
        return BEVResNetRegressionGeom(
            in_channels=mcfg.get("bev_channels", 3),
            backbone=mcfg.get("resnet", 18),
            feat_dim=mcfg.get("feat_dim", 256),
            head_hidden=mcfg.get("head_hidden", 128),
            dropout=mcfg.get("dropout", 0.1),
        )
    if name == "bev_resnet_dann":
        return BEVResNetDANN(
            in_channels=mcfg.get("bev_channels", 3),
            backbone=mcfg.get("resnet", 18),
            feat_dim=mcfg.get("feat_dim", 256),
            head_hidden=mcfg.get("head_hidden", 128),
            domain_hidden=mcfg.get("domain_hidden", 128),
            dropout=mcfg.get("dropout", 0.1),
        )
    if name == "bev_resnet_coral":
        return BEVResNetCORAL(
            in_channels=mcfg.get("bev_channels", 3),
            backbone=mcfg.get("resnet", 18),
            feat_dim=mcfg.get("feat_dim", 256),
            head_hidden=mcfg.get("head_hidden", 128),
            dropout=mcfg.get("dropout", 0.1),
        )
    # if name == "pointpillars_bev_regression":
    #     return PointPillarsBEVRegression(
    #         x_range=mcfg.get("x_range", (-1.5, 0.5)),
    #         y_range=mcfg.get("y_range", (-2.0, 2.0)),
    #         z_range=mcfg.get("z_range", (-2.0, 2.0)),
    #         res=mcfg.get("res", 0.033),
    #         joint_shift_x=mcfg.get("joint_shift_x", 0.0),
    #         point_feat_dim=mcfg.get("point_feat_dim", 64),
    #         pillar_feat_dim=mcfg.get("pillar_feat_dim", 64),
    #         backbone=mcfg.get("resnet", 18),
    #         feat_dim=mcfg.get("feat_dim", 256),
    #         head_hidden=mcfg.get("head_hidden", 128),
    #         dropout=mcfg.get("dropout", 0.1),
    #     )
    # if name == "hitch_query_transformer":
    #     return HitchQueryTransformer(
    #         bev_in_channels=mcfg.get("bev_channels", 3),
    #         bev_backbone=mcfg.get("resnet", 18),
    #         d_model=mcfg.get("d_model", 256),
    #         num_queries=mcfg.get("num_queries", 4),
    #         num_heads=mcfg.get("num_heads", 4),
    #         mem_len=mcfg.get("mem_len", 4),
    #         use_ref_points=mcfg.get("use_ref_points", True),
    #         use_mem_gate=mcfg.get("use_mem_gate", False),
    #         use_deformable=mcfg.get("use_deformable", False),
    #         num_points=mcfg.get("num_points", 4),
    #         head_hidden=mcfg.get("head_hidden", 128),
    #         dropout=mcfg.get("dropout", 0.1),
    #         head_pool=mcfg.get("head_pool", "mean"),
    #         use_2d_pe=mcfg.get("use_2d_pe", True),
    #         use_centroid_init=mcfg.get("use_centroid_init", True),
    #         use_centroid_ref=mcfg.get("use_centroid_ref", False),
    #         use_joint_pe=mcfg.get("use_joint_pe", False),
    #         pe_x_range=mcfg.get("pe_x_range"),
    #         pe_y_range=mcfg.get("pe_y_range"),
    #         pe_norm=mcfg.get("pe_norm", "01"),
    #         ref_offset_scale=mcfg.get("ref_offset_scale", 0.05),
    #     )

    # if name == "imu_gru":
    #     return IMUGRU(
    #         imu_dim=mcfg.get("imu_dim", 3),
    #         hidden_dim=mcfg.get("hidden_dim", 128),
    #         num_layers=mcfg.get("num_layers", 1),
    #         dropout=mcfg.get("dropout", 0.0),
    #         head_hidden=mcfg.get("head_hidden", 128),
    #     )

    # if name == "bev_imu_fusion":
    #     return BEVIMUFusion(
    #         bev_in_channels=mcfg.get("bev_channels", 3),
    #         bev_backbone=mcfg.get("resnet", 18),
    #         bev_feat_dim=mcfg.get("bev_feat_dim", 256),
    #         imu_dim=mcfg.get("imu_dim", 3),
    #         imu_hidden_dim=mcfg.get("imu_hidden_dim", 128),
    #         imu_layers=mcfg.get("imu_layers", 1),
    #         fusion=mcfg.get("fusion", "late"),  # 'late' or 'gated'
    #         head_hidden=mcfg.get("head_hidden", 128),
    #         dropout=mcfg.get("dropout", 0.1),
    #         bev_occ_mask=mcfg.get("bev_occ_mask", False),
    #     )

    # if name == "bev_imu_fusion_regression":
    #     return BEVIMUFusionRegression(
    #         bev_in_channels=mcfg.get("bev_channels", 3),
    #         bev_backbone=mcfg.get("resnet", 18),
    #         bev_feat_dim=mcfg.get("bev_feat_dim", 256),
    #         imu_dim=mcfg.get("imu_dim", 3),
    #         imu_hidden_dim=mcfg.get("imu_hidden_dim", 128),
    #         imu_layers=mcfg.get("imu_layers", 1),
    #         fusion=mcfg.get("fusion", "late"),
    #         head_hidden=mcfg.get("head_hidden", 128),
    #         dropout=mcfg.get("dropout", 0.1),
    #         bev_occ_mask=mcfg.get("bev_occ_mask", False),
    #     )

    # if name == "bev_imu_fusion_residual":
    #     return BEVIMUFusionResidual(
    #         bev_in_channels=mcfg.get("bev_channels", 3),
    #         bev_backbone=mcfg.get("resnet", 18),
    #         bev_feat_dim=mcfg.get("bev_feat_dim", 256),
    #         imu_dim=mcfg.get("imu_dim", 3),
    #         imu_hidden_dim=mcfg.get("imu_hidden_dim", 128),
    #         imu_layers=mcfg.get("imu_layers", 1),
    #         fusion=mcfg.get("fusion", "late"),
    #         head_hidden=mcfg.get("head_hidden", 128),
    #         dropout=mcfg.get("dropout", 0.1),
    #         delta_max_deg=mcfg.get("delta_max_deg", 30.0),
    #     )

    if name == "rule_based":
        return RuleBasedCentroid()
    if name == "rule_based_pca":
        return RuleBasedPCA()
    if name == "rule_based_ols":
        return RuleBasedOLS()
    if name == "rule_based_mle":
        return RuleBasedMLE()

    raise ValueError(f"Unknown model name: {name}")
