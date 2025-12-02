#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hitchnet.py

Multi-scale temporal + graph-attention spatial HitchNet

- Temporal:
    Micro GRU (intra-frame)  : (B, T, M, 5) → (B, T, H_micro)
    Macro Transformer + ROPE : (B, T, H_micro) → (B, H_macro)

- Spatial:
    Point MLP + kNN-based GAT : (B, N, 3) → (B, N, H_spat)

- Fusion:
    Temporal global feature as Q
    Spatial per-point tokens as K, V
    → Cross-attention → fused representation → regression head → (cos, sin)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------
# 0) Rotary Positional Embedding (ROPE)
# ---------------------------------------------------------
class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        # dim은 짝수라고 가정
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)  # [dim/2]

    def forward(self, seq_len: int, device=None):
        if device is None:
            device = self.inv_freq.device
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)  # [T]
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # [T, dim/2]
        return torch.cat([freqs.sin(), freqs.cos()], dim=-1)  # [T, dim]

    @staticmethod
    def apply(x: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
        """
        x:    [B, T, D]
        rope: [T, D] or [1, T, D]
        """
        if rope.dim() == 2:
            rope = rope.unsqueeze(0)  # [1, T, D]

        x1, x2 = x[..., 0::2], x[..., 1::2]          # [B, T, D/2]
        sin, cos = rope[..., 0::2], rope[..., 1::2]  # [1, T, D/2]

        x_rot_1 = x1 * cos - x2 * sin
        x_rot_2 = x1 * sin + x2 * cos

        out = torch.empty_like(x)
        out[..., 0::2] = x_rot_1
        out[..., 1::2] = x_rot_2
        return out


# ---------------------------------------------------------
# 1) Micro-level GRU: (B, T, M, C_in) -> (B, T, H_micro)
# ---------------------------------------------------------
class MicroGRUEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 5,     # imu(3) + vel(1) + steer(1)
        hidden_dim: int = 64,
        num_layers: int = 1,
        bidirectional: bool = False,
    ):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, M, C_in)
        return: (B, T, H_out)
        """
        B, T, M, C = x.shape
        x = x.reshape(B * T, M, C)  # (B*T, M, C)

        _, h_n = self.gru(x)       # h_n: (L * D, B*T, H)
        if self.bidirectional:
            h_last = torch.cat([h_n[-2], h_n[-1]], dim=-1)  # (B*T, 2H)
        else:
            h_last = h_n[-1]       # (B*T, H)

        frame_feat = h_last.reshape(B, T, -1)  # (B, T, H_out)
        return frame_feat


# ---------------------------------------------------------
# 2) Macro-level Transformer-ROPE: (B, T, H_micro) -> (B, H_macro)
# ---------------------------------------------------------
class MacroTransformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 64,
        model_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_rope: bool = True,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.use_rope = use_rope

        self.proj = nn.Linear(input_dim, model_dim)
        if use_rope:
            self.rope = RotaryEmbedding(model_dim)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.out_dim = model_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, C_in)
        return: (B, H_out)
        """
        # 1) projection
        z = self.proj(x)  # (B, T, D)

        # 2) ROPE
        if self.use_rope:
            rope_emb = self.rope(z.size(1), z.device)  # (T, D)
            z = RotaryEmbedding.apply(z, rope_emb)     # (B, T, D)

        # 3) Transformer encoder (self-attention 포함)
        h = self.encoder(z)  # (B, T, D)

        # 4) pooling
        out = h[:, -1, :]

        return out  # (B, D)


# ---------------------------------------------------------
# 3) Graph Attention Layer (k-NN 기반 GAT)
# ---------------------------------------------------------
class GraphAttentionLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        heads: int = 4,
        k: int = 16,
        dropout: float = 0.0,
        alpha: float = 0.2,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.k = k
        self.dropout = nn.Dropout(dropout)

        self.W = nn.Linear(in_dim, out_dim * heads, bias=False)
        self.a = nn.Linear(2 * out_dim, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x:    (B, N, C_in)
        mask: (B, N)  # padded=False, valid=True
        return: (B, N, heads * out_dim)
        """
        B, N, C = x.shape

        # 1) Linear projection + reshape heads
        h = self.W(x)                                # (B, N, H*D)
        h = h.view(B, N, self.heads, self.out_dim)   # (B, N, H, D)

        # --------------------------------------------
        # 2) Mask-aware kNN
        # --------------------------------------------
        with torch.no_grad():
            valid_pair_mask = mask.unsqueeze(1) & mask.unsqueeze(2)  # (B, N, N)

            dist = torch.cdist(x, x)                                 # (B, N, N)
            dist = dist.masked_fill(~valid_pair_mask, 1e9)

            knn_idx = dist.topk(self.k + 1, largest=False).indices[:, :, 1:]
            # (B, N, k)

        # --------------------------------------------
        # 3) Gather neighbor features
        # --------------------------------------------
        # h_expanded: (B, N, N, H, D)
        h_expanded = h.unsqueeze(1).expand(-1, N, -1, -1, -1)

        # knn_idx: (B, N, k) → (B, N, k, 1, 1) → (B, N, k, H, D)
        knn_idx_exp = knn_idx.unsqueeze(-1).unsqueeze(-1)
        knn_idx_exp = knn_idx_exp.expand(-1, -1, -1, self.heads, self.out_dim)

        h_neigh = torch.gather(h_expanded, 2, knn_idx_exp)           # (B, N, k, H, D)

        # --------------------------------------------
        # 4) Attention coefficients
        # --------------------------------------------
        h_center = h.unsqueeze(2).expand(-1, -1, self.k, -1, -1)     # (B, N, k, H, D)

        attn_input = torch.cat([h_center, h_neigh], dim=-1)          # (B, N, k, H, 2D)
        e = self.leakyrelu(self.a(attn_input).squeeze(-1))           # (B, N, k, H)

        alpha = F.softmax(e, dim=2)                                  # (B, N, k, H)
        alpha = self.dropout(alpha)

        # --------------------------------------------
        # 5) Weighted sum
        # --------------------------------------------
        alpha_exp = alpha.unsqueeze(-1)                              # (B, N, k, H, 1)
        out = (alpha_exp * h_neigh).sum(dim=2)                       # (B, N, H, D)

        out = out.reshape(B, N, self.heads * self.out_dim)           # (B, N, H*D)
        return out


# ---------------------------------------------------------
# 4) Point GAT Encoder: PointCloud -> per-point features
# ---------------------------------------------------------
class PointGATEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int = 3,
        hidden_dim: int = 128,
        layers: int = 3,
        heads: int = 4,
        k: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        gat_layers = []
        for _ in range(layers):
            gat_layers.append(
                GraphAttentionLayer(
                    in_dim=hidden_dim,
                    out_dim=hidden_dim // heads,
                    heads=heads,
                    k=k,
                    dropout=dropout,
                )
            )
        self.gat_layers = nn.ModuleList(gat_layers)
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(layers)])

    def forward(self, pcd: torch.Tensor, mask: torch.Tensor):
        """
        pcd:  (B, N, 3)
        mask: (B, N)
        return:
          - per-point feats: (B, N, hidden_dim)
          - global max-pooled: (B, hidden_dim)
        """
        x = self.input_mlp(pcd)  # (B, N, hidden_dim)
        for gat, ln in zip(self.gat_layers, self.norms):
            residual = x
            x = gat(x, mask)     # (B, N, hidden_dim)
            x = ln(x + residual)
            x = F.relu(x)

        global_feat = x.max(dim=1).values  # (B, hidden_dim)
        return x, global_feat


# ---------------------------------------------------------
# 5) Fusion block: temporal feature Q vs spatial tokens K,V
# ---------------------------------------------------------
class FusionBlock(nn.Module):
    def __init__(
        self,
        temp_dim: int,
        spat_dim: int,
        fusion_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
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

    def forward(self, temp_feat: torch.Tensor, spat_tokens: torch.Tensor) -> torch.Tensor:
        """
        temp_feat:   (B, temp_dim)        # macro Transformer output
        spat_tokens: (B, N, spat_dim)    # per-point features from GAT
        return:      (B, fusion_dim)
        """
        B, N, _ = spat_tokens.shape

        Q = self.temp_proj(temp_feat).unsqueeze(1)  # (B, 1, F)
        K = self.spat_proj(spat_tokens)             # (B, N, F)
        V = K

        attn_out, _ = self.mha(Q, K, V)            # (B, 1, F)
        out = attn_out.squeeze(1)                  # (B, F)

        # residual + norm
        out = self.norm(out + self.temp_proj(temp_feat))
        out = self.dropout(out)
        return out  # (B, F)


# ---------------------------------------------------------
# 6) HitchNet: 전체 모델 (Micro-GRU + Macro-Transformer + GAT + cross-attn)
# ---------------------------------------------------------
class HitchNet(nn.Module):
    def __init__(
        self,
        micro_input_dim: int = 5,       # imu(3) + vel(1) + steer(1)
        micro_hidden_dim: int = 64,
        macro_hidden_dim: int = 128,
        micro_layers: int = 1,
        macro_layers: int = 2,          # Transformer layer 수
        bidirectional_micro: bool = False,
        gat_hidden_dim: int = 128,
        gat_layers: int = 3,
        gat_heads: int = 4,
        gat_k: int = 16,
        fusion_dim: int = 256,
        fusion_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Micro GRU (intra-frame)
        self.micro_encoder = MicroGRUEncoder(
            input_dim=micro_input_dim,
            hidden_dim=micro_hidden_dim,
            num_layers=micro_layers,
            bidirectional=bidirectional_micro,
        )
        micro_out_dim = micro_hidden_dim * (2 if bidirectional_micro else 1)

        # Macro Transformer-ROPE (inter-frame)
        self.macro_encoder = MacroTransformerEncoder(
            input_dim=micro_out_dim,
            model_dim=macro_hidden_dim,
            num_layers=macro_layers,
            num_heads=fusion_heads,
            dropout=dropout,
            use_rope=True,
        )
        macro_out_dim = self.macro_encoder.out_dim  # = macro_hidden_dim

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

    def forward(self, batch: dict) -> torch.Tensor:
        """
        batch: dict with keys
        - pcd:       (B, N, 3)
        - pcd_mask:  (B, N)
        - imu:       (B, T, M, 3)
        - velocity:  (B, T, M, 1)
        - steering:  (B, T, M, 1)

        return:
        - pred: (B, 2)  # (cos, sin)
        """
        pcd = batch["pcd"]           # (B, N, 3)
        mask = batch["pcd_mask"]     # (B, N)

        imu = batch["imu"]           # (B, T, M, 3)
        vel = batch["velocity"]      # (B, T, M, 1)
        steer = batch["steering"]    # (B, T, M, 1)

        # -----------------------------
        # 1) Temporal Encoder
        # -----------------------------
        temporal_in = torch.cat([imu, vel, steer], dim=-1)   # (B, T, M, 5)
        frame_feats = self.micro_encoder(temporal_in)        # (B, T, H_micro)
        temporal_feat = self.macro_encoder(frame_feats)      # (B, H_macro)

        # -----------------------------
        # 2) Spatial Encoder
        # -----------------------------
        spat_tokens, _ = self.point_encoder(pcd, mask)       # (B, N, gat_hidden_dim)

        # -----------------------------
        # 3) Fusion (temporal Q, spatial K/V)
        # -----------------------------
        fused = self.fusion(temporal_feat, spat_tokens)      # (B, fusion_dim)

        # -----------------------------
        # 4) Regression Head
        # -----------------------------
        pred = self.head(fused)                              # (B, 2) → (cos, sin)

        return pred
