import torch
import torch.nn as nn


class EdgeTransformer(nn.Module):
    """
    Transformer 模块，用于处理节点对的交互建模。
    输入：每条边上的两个节点嵌入，形状为 (batch_size, 2, hidden_dim)
    输出：每条边的表示，形状为 (batch_size, hidden_dim)
    """
    def __init__(self, hidden_dim: int, num_heads: int = 4, num_layers: int = 3, dropout: float = 0.4):
        super(EdgeTransformer, self).__init__()
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
        """
        edge_embed: Tensor of shape (batch_size, 2, hidden_dim)
        """
        x = self.transformer(edge_embed)               # shape: (batch_size, 2, hidden_dim)
        x = x.mean(dim=1)                              # 可替代为 x[:, 0] 也行
        return self.output_proj(x)

