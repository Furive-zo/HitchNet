import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------
# 1) Micro-level GRU: (B, T, M, C_in) -> (B, T, H_micro)
# ---------------------------------------------------------
class MicroGRUEncoder(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, num_layers=1, bidirectional=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

    def forward(self, x):
        """
        x: (B, T, M, C_in)
        return: (B, T, H_out)
        """
        B, T, M, C = x.shape
        x = x.reshape(B * T, M, C)      # (B*T, M, C)

        _, h_n = self.gru(x)           # h_n: (num_layers * num_directions, B*T, H)
        if self.bidirectional:
            # concat last layer forward/backward
            h_last = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            h_last = h_n[-1]           # (B*T, H)

        frame_feat = h_last.reshape(B, T, -1)  # (B, T, H_out)
        return frame_feat


# ---------------------------------------------------------
# 2) Macro-level GRU: (B, T, H_micro_out) -> (B, H_macro)
# ---------------------------------------------------------
class MacroGRUEncoder(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128, num_layers=1, bidirectional=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

    def forward(self, x):
        """
        x: (B, T, C_in)
        return: (B, H_out)  # 마지막 프레임 시점의 macro feature
        """
        # x: (B, T, C_in)
        _, h_n = self.gru(x)  # h_n: (L * D, B, H)
        if self.bidirectional:
            h_last = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            h_last = h_n[-1]
        return h_last  # (B, H_out)


# ---------------------------------------------------------
# 3) Graph Attention Layer (k-NN 기반 GAT)
# ---------------------------------------------------------
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, heads=4, k=16, dropout=0.0, alpha=0.2):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.k = k
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout)

        self.W = nn.Linear(in_dim, out_dim * heads, bias=False)
        self.a = nn.Linear(2 * out_dim, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, x, mask):
        """
        x: (B, N, C_in)
        mask: (B, N)  # padded=False, valid=True
        return: (B, N, heads * out_dim)
        """
        B, N, C = x.shape

        # 1) Linear projection + reshape heads
        h = self.W(x)                      # (B, N, H*D)
        h = h.view(B, N, self.heads, self.out_dim)   # (B, N, H, D)

        # --------------------------------------------
        # 2) Mask-aware kNN
        # --------------------------------------------
        with torch.no_grad():
            # mask → (B,1,N) & (B,N,1) → (B,N,N)
            valid_pair_mask = mask.unsqueeze(1) & mask.unsqueeze(2)   # valid only

            # distance matrix
            dist = torch.cdist(x, x)                                 # (B, N, N)

            # padded positions → large distance
            dist = dist.masked_fill(~valid_pair_mask, 1e9)

            # compute kNN index
            knn_idx = dist.topk(self.k + 1, largest=False).indices[:, :, 1:]  
            # (B, N, k)

        # --------------------------------------------
        # 3) Gather neighbor features
        # --------------------------------------------
        # h_expanded: (B, N, N, H, D)
        h_expanded = h.unsqueeze(1).expand(-1, N, -1, -1, -1)

        # knn_idx: (B,N,k) → (B,N,k,1,1) → expand -> (B,N,k,H,D)
        knn_idx_exp = knn_idx.unsqueeze(-1).unsqueeze(-1)
        knn_idx_exp = knn_idx_exp.expand(-1, -1, -1, self.heads, self.out_dim)

        # gather neighbors: dim=2
        h_neigh = torch.gather(h_expanded, 2, knn_idx_exp)           # (B,N,k,H,D)

        # --------------------------------------------
        # 4) Attention coefficients
        # --------------------------------------------
        # center feats
        h_center = h.unsqueeze(2).expand(-1, -1, self.k, -1, -1)     # (B,N,k,H,D)

        attn_input = torch.cat([h_center, h_neigh], dim=-1)          # (B,N,k,H,2D)
        e = self.leakyrelu(self.a(attn_input).squeeze(-1))           # (B,N,k,H)

        # softmax along neighbor dimension
        alpha = F.softmax(e, dim=2)                                 # (B,N,k,H)
        alpha = self.dropout(alpha)

        # --------------------------------------------
        # 5) Weighted sum
        # --------------------------------------------
        alpha_exp = alpha.unsqueeze(-1)                              # (B,N,k,H,1)
        out = (alpha_exp * h_neigh).sum(dim=2)                       # (B,N,H,D)

        out = out.reshape(B, N, self.heads * self.out_dim)           # (B,N,H*D)
        return out



# ---------------------------------------------------------
# 4) Point GAT Encoder: PointCloud -> per-point features
# ---------------------------------------------------------
class PointGATEncoder(nn.Module):
    def __init__(self, in_dim=3, hidden_dim=128, layers=3, heads=4, k=16, dropout=0.1):
        super().__init__()
        self.input_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        gat_layers = []
        dim = hidden_dim
        for _ in range(layers):
            gat_layers.append(
                GraphAttentionLayer(
                    in_dim=dim,
                    out_dim=dim // heads,
                    heads=heads,
                    k=k,
                    dropout=dropout,
                )
            )
            dim = dim  # heads * out_dim = hidden_dim 유지
        self.gat_layers = nn.ModuleList(gat_layers)
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, pcd, mask):
        x = self.input_mlp(pcd)
        for gat, ln in zip(self.gat_layers, self.norms):
            residual = x
            x = gat(x, mask)    # ← mask 전달
            x = ln(x + residual)
            x = F.relu(x)
        global_feat = x.max(dim=1).values
        return x, global_feat

# ---------------------------------------------------------
# 5) Fusion block: temporal feature Q vs spatial tokens K,V
# ---------------------------------------------------------
class FusionBlock(nn.Module):
    def __init__(self, temp_dim, spat_dim, fusion_dim=256, num_heads=4, dropout=0.1):
        super().__init__()
        self.temp_proj = nn.Linear(temp_dim, fusion_dim)
        self.spat_proj = nn.Linear(spat_dim, fusion_dim)
        self.mha = nn.MultiheadAttention(
            fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(fusion_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, temp_feat, spat_tokens):
        """
        temp_feat: (B, temp_dim)          # macro GRU output
        spat_tokens: (B, N, spat_dim)    # per-point features from GAT
        """
        B, N, _ = spat_tokens.shape

        Q = self.temp_proj(temp_feat).unsqueeze(1)  # (B, 1, F)
        K = self.spat_proj(spat_tokens)            # (B, N, F)
        V = K

        attn_out, _ = self.mha(Q, K, V)           # (B, 1, F)
        out = attn_out.squeeze(1)                 # (B, F)

        # residual + norm
        out = self.norm(out + self.temp_proj(temp_feat))
        out = self.dropout(out)
        return out  # (B, F)


# ---------------------------------------------------------
# 6) HitchNet: 전체 모델
# ---------------------------------------------------------
class HitchNet(nn.Module):
    def __init__(
        self,
        micro_input_dim=5,       # imu(3) + vel(1) + steer(1)
        micro_hidden_dim=64,
        macro_hidden_dim=128,
        micro_layers=1,
        macro_layers=1,
        bidirectional_micro=False,
        bidirectional_macro=False,
        gat_hidden_dim=128,
        gat_layers=3,
        gat_heads=4,
        gat_k=16,
        fusion_dim=256,
        fusion_heads=4,
        dropout=0.1,
    ):
        super().__init__()

        # Micro GRU
        self.micro_encoder = MicroGRUEncoder(
            input_dim=micro_input_dim,
            hidden_dim=micro_hidden_dim,
            num_layers=micro_layers,
            bidirectional=bidirectional_micro,
        )
        micro_out_dim = micro_hidden_dim * (2 if bidirectional_micro else 1)

        # Macro GRU
        self.macro_encoder = MacroGRUEncoder(
            input_dim=micro_out_dim,
            hidden_dim=macro_hidden_dim,
            num_layers=macro_layers,
            bidirectional=bidirectional_macro,
        )
        macro_out_dim = macro_hidden_dim * (2 if bidirectional_macro else 1)

        # Point GAT
        self.point_encoder = PointGATEncoder(
            in_dim=3,
            hidden_dim=gat_hidden_dim,
            layers=gat_layers,
            heads=gat_heads,
            k=gat_k,
            dropout=dropout,
        )

        # Fusion block
        self.fusion = FusionBlock(
            temp_dim=macro_out_dim,
            spat_dim=gat_hidden_dim,
            fusion_dim=fusion_dim,
            num_heads=fusion_heads,
            dropout=dropout,
        )

        # Regression head: [cos, sin]
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, 2),
        )

    def forward(self, batch):
        """
        batch: dict with keys
        - pcd: (B, N, 3)
        - imu: (B, T, M, 3)
        - velocity: (B, T, M, 1)
        - steering: (B, T, M, 1)

        return:
        - pred: (B, 2)
        """

        pcd = batch["pcd"]            # (B, N, 3)
        imu = batch["imu"]            # (B, T, M, 3)
        vel = batch["velocity"]       # (B, T, M, 1)
        steer = batch["steering"]     # (B, T, M, 1)
        mask = batch["pcd_mask"]

        # -----------------------------
        # 1) Temporal Encoder
        # -----------------------------
        temporal_in = torch.cat([imu, vel, steer], dim=-1)   # (B,T,M,5)
        frame_feats = self.micro_encoder(temporal_in)        # (B,T,H_micro)
        temporal_feat = self.macro_encoder(frame_feats)      # (B,H_macro)

        # -----------------------------
        # 2) Spatial Encoder
        # -----------------------------
        spat_tokens, _ = self.point_encoder(pcd, mask)

        # -----------------------------
        # 3) Fusion (temporal Q, spatial K/V)
        # -----------------------------
        fused = self.fusion(temporal_feat, spat_tokens)      # (B,F)

        # -----------------------------
        # 4) Regression Head
        # -----------------------------
        pred = self.head(fused)                              # (B,2)

        return pred

