# models/builder.py

from .hitchnet import HitchNet

def build_hitchnet(mcfg):
    tcfg = mcfg["temporal"]
    scfg = mcfg["spatial"]
    fcfg = mcfg["fusion"]

    model = HitchNet(
        # --------------------
        # Temporal
        # --------------------
        micro_input_dim=tcfg.get("micro_input_dim", 5),
        micro_hidden_dim=tcfg.get("micro_hidden_dim", 64),
        micro_layers=tcfg.get("micro_layers", 1),
        bidirectional_micro=tcfg.get("bidirectional_micro", False),

        macro_hidden_dim=tcfg.get("macro_hidden_dim", 128),
        macro_layers=tcfg.get("macro_layers", 2),

        # --------------------
        # Spatial
        # --------------------
        gat_hidden_dim=scfg.get("gat_hidden_dim", 128),
        gat_layers=scfg.get("gat_layers", 3),
        gat_heads=scfg.get("gat_heads", 4),
        gat_k=scfg.get("gat_k", 16),

        # --------------------
        # Fusion
        # --------------------
        fusion_dim=fcfg.get("fusion_dim", 256),
        fusion_heads=fcfg.get("fusion_heads", 4),
        dropout=fcfg.get("dropout", 0.1),
    )

    return model
