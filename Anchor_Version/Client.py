import torch
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling


class Client:
    def __init__(self, client_id, data, encoder, decoder,
                 anchor_list=None, peer_z=None,
                 contrastive_weight=1.0, device='cpu', lr=0.005, weight_decay=1e-4):
        self.client_id = client_id
        self.data = data.to(device)
        self.device = device
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.anchor_list = anchor_list
        self.peer_z = peer_z  # 同一轮图1或图2传来的锚点嵌入（图外共享）
        self.contrastive_weight = contrastive_weight

        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=lr,
            weight_decay=weight_decay
        )
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def train(self):
        self.encoder.train()
        self.decoder.train()
        self.optimizer.zero_grad()

        pos_edge_index = self.data.edge_index
        neg_edge_index = negative_sampling(
            edge_index=pos_edge_index,
            num_nodes=self.data.num_nodes,
            num_neg_samples=pos_edge_index.size(1)
        )

        z = self.encoder(self.data.x, self.data.edge_index)

        pos_pred = self.decoder(z[pos_edge_index[0]], z[pos_edge_index[1]])
        neg_pred = self.decoder(z[neg_edge_index[0]], z[neg_edge_index[1]])

        labels = torch.cat([
            torch.ones(pos_pred.size(0), device=self.device),
            torch.zeros(neg_pred.size(0), device=self.device)
        ])
        pred = torch.cat([pos_pred, neg_pred], dim=0).squeeze()

        link_loss = self.criterion(pred, labels)

        # === 对比损失（如果 anchor_list 和 peer_z 存在） ===
        contrastive_loss = 0
        if self.anchor_list is not None and self.peer_z is not None:
            src_ids = [a[0] for a in self.anchor_list]
            tgt_ids = [a[1] for a in self.anchor_list]
            if self.client_id == 0:
                # 图1：我们保留 anchor 节点编号为 src
                anchor_z = z[src_ids]
                peer_anchor_z = self.peer_z[tgt_ids]
            else:
                # 图2：我们保留 anchor 节点编号为 tgt
                anchor_z = z[tgt_ids]
                peer_anchor_z = self.peer_z[src_ids]

            contrastive_loss = F.mse_loss(anchor_z, peer_anchor_z)

        total_loss = link_loss + self.contrastive_weight * contrastive_loss
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()

    def evaluate(self, use_test=False):
        self.encoder.eval()
        self.decoder.eval()

        with torch.no_grad():
            z = self.encoder(self.data.x, self.data.edge_index)

            if use_test:
                pos_edge_index = self.data.test_pos_edge_index
                neg_edge_index = self.data.test_neg_edge_index
            else:
                pos_edge_index = self.data.val_pos_edge_index
                neg_edge_index = self.data.val_neg_edge_index

            pos_pred = self.decoder(z[pos_edge_index[0]], z[pos_edge_index[1]])
            neg_pred = self.decoder(z[neg_edge_index[0]], z[neg_edge_index[1]])

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

        return acc, recall, precision, f1, z  # 现在也返回嵌入用于对比学习

    def get_encoder_state(self):
        return self.encoder.state_dict()

    def get_decoder_state(self):
        return self.decoder.state_dict()

    def set_encoder_state(self, state_dict):
        self.encoder.load_state_dict(state_dict)

    def set_decoder_state(self, state_dict):
        self.decoder.load_state_dict(state_dict)
