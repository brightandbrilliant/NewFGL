import torch
import torch.nn as nn


class EdgeTransformer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int = 4, num_layers: int = 3, dropout: float = 0.4):
        super(EdgeTransformer, self).__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, edge_embed: torch.Tensor):
        # edge_embed: (batch_size, 2, input_dim)
        x = self.input_proj(edge_embed)               # → (batch_size, 2, hidden_dim)
        x = self.transformer(x)                       # → (batch_size, 2, hidden_dim)
        x = x.mean(dim=1)                             # 聚合节点对
        return self.output_proj(x)                    # 输出最终边表示
