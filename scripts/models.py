import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, EdgeConv

# ----- Spatial Blocks -----

class DGCNNBlock(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.edge_conv = EdgeConv(
            nn=nn.Sequential(
                nn.Linear(2 * input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, input_dim)
            )
        )

    def forward(self, x, edge_index):
        return self.edge_conv(x, edge_index)

# ----- Temporal Blocks -----

class TemporalTCN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.tcn1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2)
        self.relu1 = nn.ReLU()
        self.tcn2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU()
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):  # x: [B, T, L]
        x = x.permute(0, 2, 1)  # [B, L, T]
        x = self.relu1(self.tcn1(x))
        x = self.relu2(self.tcn2(x))
        x = x[:, :, -1]  # [B, hidden_dim]
        return self.linear(x)

class TemporalLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):  # x: [B, T, L]
        out, _ = self.lstm(x)
        return self.linear(out[:, -1, :])  # 마지막 타임스텝

class TemporalTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, nhead=4, num_layers=2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.linear = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):  # x: [B, T, L]
        out = self.transformer(x)
        return self.linear(out[:, -1, :])

class TemporalMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim)
        )

    def forward(self, x):  # x: [B, T, L]
        return self.mlp(x[:, -1, :])
    
# ----- Fusion Blocks -----

class GatingFusion(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.gate(x)

class CrossAttentionFusion(nn.Module):
    def __init__(self, graph_dim, temporal_dim, fusion_output_dim):
        super().__init__()
        self.graph_proj = nn.Linear(graph_dim, fusion_output_dim)
        self.temporal_proj = nn.Linear(temporal_dim, fusion_output_dim)

        self.cross1 = nn.MultiheadAttention(fusion_output_dim, 4, batch_first=True)
        self.cross2 = nn.MultiheadAttention(fusion_output_dim, 4, batch_first=True)
        self.linear = nn.Linear(fusion_output_dim, fusion_output_dim)

    def forward(self, graph_feat, temporal_feat):  # [B, D1], [B, D2]
        g = self.graph_proj(graph_feat).unsqueeze(1)     # [B, 1, D]
        t = self.temporal_proj(temporal_feat).unsqueeze(1)  # [B, 1, D]

        attn1, _ = self.cross1(g, t, t)
        attn2, _ = self.cross2(t, g, g)

        fused = attn1 + attn2  # [B, 1, D]
        return self.linear(fused.squeeze(1))  # [B, D]
    
class GatingCrossAttentionFusion(nn.Module):
    def __init__(self, graph_dim, temporal_dim, fusion_output_dim):
        super().__init__()
        self.graph_proj = nn.Linear(graph_dim, fusion_output_dim)
        self.temporal_proj = nn.Linear(temporal_dim, fusion_output_dim)

        self.cross1 = nn.MultiheadAttention(fusion_output_dim, 4, batch_first=True)
        self.cross2 = nn.MultiheadAttention(fusion_output_dim, 4, batch_first=True)

        self.gate = nn.Sequential(
            nn.Linear(fusion_output_dim * 2, fusion_output_dim),
            nn.Sigmoid()
        )

        self.linear = nn.Linear(fusion_output_dim, fusion_output_dim)

    def forward(self, graph_feat, temporal_feat):  # [B, D1], [B, D2]
        g = self.graph_proj(graph_feat).unsqueeze(1)  # [B, 1, D]
        t = self.temporal_proj(temporal_feat).unsqueeze(1)  # [B, 1, D]

        attn1, _ = self.cross1(g, t, t)  # spatial attends to temporal
        attn2, _ = self.cross2(t, g, g)  # temporal attends to spatial

        cross = attn1 + attn2  # [B, 1, D]
        cross = cross.squeeze(1)  # [B, D]

        concat = torch.cat([g.squeeze(1), cross], dim=-1)  # [B, 2D]
        gate = self.gate(concat)  # [B, D]

        fused = gate * g.squeeze(1) + (1 - gate) * cross  # gated fusion
        return self.linear(fused)  # [B, D]
    

class GatingCrossAttentionFusionResidual(nn.Module):
    def __init__(self, graph_dim, temporal_dim, fusion_output_dim):
        super().__init__()
        self.graph_proj = nn.Linear(graph_dim, fusion_output_dim)
        self.temporal_proj = nn.Linear(temporal_dim, fusion_output_dim)

        self.cross1 = nn.MultiheadAttention(fusion_output_dim, 4, batch_first=True)
        self.cross2 = nn.MultiheadAttention(fusion_output_dim, 4, batch_first=True)

        self.gate = nn.Sequential(
            nn.Linear(fusion_output_dim * 2, fusion_output_dim),
            nn.Sigmoid()
        )

        self.norm = nn.LayerNorm(fusion_output_dim)
        self.linear = nn.Linear(fusion_output_dim, fusion_output_dim)

    def forward(self, graph_feat, temporal_feat):  # [B, D1], [B, D2]
        g = self.graph_proj(graph_feat).unsqueeze(1)  # [B, 1, D]
        t = self.temporal_proj(temporal_feat).unsqueeze(1)  # [B, 1, D]

        attn1, _ = self.cross1(g, t, t)  # [B, 1, D]
        attn2, _ = self.cross2(t, g, g)  # [B, 1, D]

        cross = attn1 + attn2  # [B, 1, D]
        cross = cross.squeeze(1)  # [B, D]
        g = g.squeeze(1)  # [B, D]

        gate = self.gate(torch.cat([g, cross], dim=-1))  # [B, D]
        fused = gate * g + (1 - gate) * cross  # [B, D]

        # Residual + LayerNorm + Linear
        fused = self.norm(fused + g)  # [B, D]
        return self.linear(fused)
    
# ----- Main Model -----

class GATSpatialTemporal(nn.Module):
    def __init__(self,
                 graph_input_dim,
                 graph_hidden_dim,
                 temporal_hidden_dim,
                 num_heads,
                 output_dim,
                 temporal_type='TCN',
                 fusion_type='gating',
                 use_dgcnn=False):
        super().__init__()
        self.use_dgcnn = use_dgcnn

        if self.use_dgcnn:
            self.dgcnn = DGCNNBlock(graph_input_dim)
            gat_input_dim = graph_input_dim
        else:
            gat_input_dim = graph_input_dim

        self.gat1 = GATConv(gat_input_dim, graph_hidden_dim, heads=num_heads, concat=True)
        self.gat2 = GATConv(graph_hidden_dim * num_heads, graph_hidden_dim, heads=num_heads, concat=False)

        temporal_input_dim = 10 * 3

        if temporal_type == "TCN":
            self.temporal_encoder = TemporalTCN(temporal_input_dim, temporal_hidden_dim)
        elif temporal_type == "LSTM":
            self.temporal_encoder = TemporalLSTM(temporal_input_dim, temporal_hidden_dim)
        elif temporal_type == "Transformer":
            self.temporal_encoder = TemporalTransformer(temporal_input_dim, temporal_hidden_dim)
        elif temporal_type == "MLP":
            self.temporal_encoder = TemporalMLP(temporal_input_dim, temporal_hidden_dim)
        else:
            raise ValueError(f"Invalid temporal_type: {temporal_type}")

        fusion_input_dim = graph_hidden_dim + temporal_hidden_dim
        if fusion_type == 'concat':
            self.fusion = nn.Identity()
        elif fusion_type == 'gating':
            self.fusion = GatingFusion(fusion_input_dim)
        elif fusion_type == 'cross':
            self.fusion = CrossAttentionFusion(graph_hidden_dim, temporal_hidden_dim, fusion_input_dim)
        elif fusion_type == 'gating+cross':
            self.fusion = GatingCrossAttentionFusion(graph_hidden_dim, temporal_hidden_dim, fusion_input_dim)
        elif fusion_type == 'gating+cross+residual':
            self.fusion = GatingCrossAttentionFusionResidual(graph_hidden_dim, temporal_hidden_dim, fusion_input_dim)
        else:
            raise ValueError(f"Invalid fusion_type: {fusion_type}")

        self.fc_out = nn.Linear(fusion_input_dim, output_dim)

    def forward(self, graph, speed, steer, angular, batch):
        device = next(self.parameters()).device
        speed, steer, angular = speed.to(device), steer.to(device), angular.to(device)
        x = graph.x.to(device)
        edge_index = graph.edge_index.to(device)
        batch = batch.to(device)

        if self.use_dgcnn:
            x = self.dgcnn(x, edge_index)

        x = F.relu(self.gat1(x, edge_index))
        x = F.relu(self.gat2(x, edge_index))
        graph_feat = global_mean_pool(x, batch)

        temporal_input = torch.cat([speed, steer, angular], dim=-1)
        temporal_feat = self.temporal_encoder(temporal_input)

        if isinstance(self.fusion, (CrossAttentionFusion, GatingCrossAttentionFusion, GatingCrossAttentionFusionResidual)):
            fused = self.fusion(graph_feat, temporal_feat)
        else:
            combined = torch.cat([graph_feat, temporal_feat], dim=1)
            fused = self.fusion(combined)

        return self.fc_out(fused)
