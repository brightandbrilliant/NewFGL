import torch
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling


class Client:
    def __init__(self, client_id, data, encoder, decoder, device='cpu',
                 lr=0.005, weight_decay=1e-4, mu=0.01):
        self.client_id = client_id
        self.data = data.to(device)
        self.device = device
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=lr,
            weight_decay=weight_decay
        )
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.mu = mu

        # 全局模型参数缓存（FedProx用）
        self.global_encoder_state = None
        self.global_decoder_state = None

        # --- 增强边缓存 ---
        self.aug_pos_edges = []
        self.aug_neg_edges = []
        self.z_others = None  # 跨客户端的节点嵌入缓存

    def set_global_state(self, encoder_state, decoder_state):
        """下发全局参数时调用，用于FedProx正则"""
        self.global_encoder_state = {
            k: v.detach().clone() for k, v in encoder_state.items()
        }
        self.global_decoder_state = {
            k: v.detach().clone() for k, v in decoder_state.items()
        }

    def _compute_prox_reg(self):
        """FedProx 正则项"""
        prox_reg = 0.0
        if self.global_encoder_state is not None and self.global_decoder_state is not None:
            for (name, p) in self.encoder.named_parameters():
                prox_reg += ((p - self.global_encoder_state[name].to(self.device)) ** 2).sum()
            for (name, p) in self.decoder.named_parameters():
                prox_reg += ((p - self.global_decoder_state[name].to(self.device)) ** 2).sum()
        return prox_reg

    def train(self):
        """正常本地训练 + FedProx"""
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

        task_loss = self.criterion(pred, labels)
        loss = task_loss + (self.mu / 2.0) * self._compute_prox_reg()

        loss.backward()
        self.optimizer.step()
        return loss.item()

    # ============== 增强机制 ==============

    def inject_augmented_positive_edges(self, pos_edge_list, z_other):
        """注入增强正边"""
        self.aug_pos_edges.extend(pos_edge_list)
        self.z_others = z_other.to(self.device)

    def inject_augmented_negative_edges(self, neg_edge_list, z_other):
        """注入增强负边"""
        self.aug_neg_edges.extend(neg_edge_list)
        self.z_others = z_other.to(self.device)

    def train_on_augmented_positives(self):
        """在增强正边上训练 + FedProx"""
        if not self.aug_pos_edges or self.z_others is None:
            return 0.0

        self.encoder.train()
        self.decoder.train()
        self.optimizer.zero_grad()

        z = self.encoder(self.data.x, self.data.edge_index)
        aug_pred = []
        for u, v in self.aug_pos_edges:
            # u 在本地，v 在跨客户端
            aug_pred.append(self.decoder(z[u], self.z_others[v]))
        aug_pred = torch.cat(aug_pred, dim=0)

        labels = torch.ones(aug_pred.size(0), device=self.device)
        task_loss = self.criterion(aug_pred.squeeze(), labels)

        loss = task_loss + (self.mu / 2.0) * self._compute_prox_reg()

        loss.backward()
        self.optimizer.step()

        self.aug_pos_edges = []  # 用完清空
        return loss.item()

    def train_on_augmented_negatives(self):
        """在增强负边上训练 + FedProx"""
        if not self.aug_neg_edges or self.z_others is None:
            return 0.0

        self.encoder.train()
        self.decoder.train()
        self.optimizer.zero_grad()

        z = self.encoder(self.data.x, self.data.edge_index)
        aug_pred = []
        for u, v in self.aug_neg_edges:
            aug_pred.append(self.decoder(z[u], self.z_others[v]))
        aug_pred = torch.cat(aug_pred, dim=0)

        labels = torch.zeros(aug_pred.size(0), device=self.device)
        task_loss = self.criterion(aug_pred.squeeze(), labels)

        loss = task_loss + (self.mu / 2.0) * self._compute_prox_reg()

        loss.backward()
        self.optimizer.step()

        self.aug_neg_edges = []  # 用完清空
        return loss.item()

    # =====================================

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

        return acc, recall, precision, f1

    def get_encoder_state(self):
        return self.encoder.state_dict()

    def get_decoder_state(self):
        return self.decoder.state_dict()

    def set_encoder_state(self, state_dict):
        self.encoder.load_state_dict(state_dict)

    def set_decoder_state(self, state_dict):
        self.decoder.load_state_dict(state_dict)
