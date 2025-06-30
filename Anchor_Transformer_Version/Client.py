import torch
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling


class Client:
    def __init__(self, client_id, data, encoder, transformer, decoder,
                 anchor_list=None, peer_z=None,
                 contrastive_weight=1.0, device='cpu', lr=0.005, weight_decay=1e-4):
        self.client_id = client_id
        self.data = data.to(device)
        self.device = device

        self.encoder = encoder.to(device)
        self.transformer = transformer.to(device)
        self.decoder = decoder.to(device)

        self.anchor_list = anchor_list
        self.peer_z = peer_z
        self.contrastive_weight = contrastive_weight

        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) +
            list(self.transformer.parameters()) +
            list(self.decoder.parameters()),
            lr=lr,
            weight_decay=weight_decay
        )
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def train(self):
        self.encoder.train()
        self.transformer.train()
        self.decoder.train()
        self.optimizer.zero_grad()

        pos_edge_index = self.data.edge_index
        neg_edge_index = negative_sampling(
            edge_index=pos_edge_index,
            num_nodes=self.data.num_nodes,
            num_neg_samples=pos_edge_index.size(1)
        )

        z = self.encoder(self.data.x, self.data.edge_index)

        # Transformer 编码边对
        pos_edge_embed = torch.stack([z[pos_edge_index[0]], z[pos_edge_index[1]]], dim=1)
        neg_edge_embed = torch.stack([z[neg_edge_index[0]], z[neg_edge_index[1]]], dim=1)

        pos_edge_repr = self.transformer(pos_edge_embed)
        neg_edge_repr = self.transformer(neg_edge_embed)

        pos_pred = self.decoder(pos_edge_repr)
        neg_pred = self.decoder(neg_edge_repr)

        labels = torch.cat([
            torch.ones(pos_pred.size(0), device=self.device),
            torch.zeros(neg_pred.size(0), device=self.device)
        ])
        pred = torch.cat([pos_pred, neg_pred], dim=0).squeeze()

        link_loss = self.criterion(pred, labels)

        # === 对比损失 ===
        contrastive_loss = 0
        if self.anchor_list is not None and self.peer_z is not None:
            src_ids = [a[0] for a in self.anchor_list]
            tgt_ids = [a[1] for a in self.anchor_list]
            if self.client_id == 0:
                anchor_z = z[src_ids]
                peer_anchor_z = self.peer_z[tgt_ids]
            else:
                anchor_z = z[tgt_ids]
                peer_anchor_z = self.peer_z[src_ids]

            contrastive_loss = F.mse_loss(anchor_z, peer_anchor_z)

        total_loss = link_loss + self.contrastive_weight * contrastive_loss
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()

    def evaluate(self):
        self.encoder.eval()
        self.transformer.eval()
        self.decoder.eval()

        with torch.no_grad():
            z = self.encoder(self.data.x, self.data.edge_index)

            pos_edge_index = self.data.edge_index
            neg_edge_index = negative_sampling(
                edge_index=pos_edge_index,
                num_nodes=self.data.num_nodes,
                num_neg_samples=pos_edge_index.size(1)
            )

            pos_edge_embed = torch.stack([z[pos_edge_index[0]], z[pos_edge_index[1]]], dim=1)
            neg_edge_embed = torch.stack([z[neg_edge_index[0]], z[neg_edge_index[1]]], dim=1)

            pos_edge_repr = self.transformer(pos_edge_embed)
            neg_edge_repr = self.transformer(neg_edge_embed)

            pos_pred = self.decoder(pos_edge_repr)
            neg_pred = self.decoder(neg_edge_repr)

            pred = torch.cat([pos_pred, neg_pred], dim=0).squeeze()
            labels = torch.cat([
                torch.ones(pos_pred.size(0), device=self.device),
                torch.zeros(neg_pred.size(0), device=self.device)
            ])

            pred_label = (torch.sigmoid(pred) > 0.5).float()
            correct = (pred_label == labels).sum().item()
            acc = correct / labels.size(0)

            TP = ((pred_label == 1) & (labels == 1)).sum().item()
            FP = ((pred_label == 1) & (labels == 0)).sum().item()
            FN = ((pred_label == 0) & (labels == 1)).sum().item()

            recall = TP / (TP + FN + 1e-8)
            precision = TP / (TP + FP + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)

        return acc, recall, precision, f1, z  # 返回嵌入 z 供主程序设置 peer_z 使用

    def get_encoder_state(self):
        return self.encoder.state_dict()

    def get_decoder_state(self):
        return self.decoder.state_dict()

    def get_transformer_state(self):
        return self.transformer.state_dict()

    def set_encoder_state(self, state_dict):
        self.encoder.load_state_dict(state_dict)

    def set_decoder_state(self, state_dict):
        self.decoder.load_state_dict(state_dict)

    def set_transformer_state(self, state_dict):
        self.transformer.load_state_dict(state_dict)
