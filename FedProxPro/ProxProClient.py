import torch
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling
from collections import defaultdict


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

        # FedProx 全局参数缓存
        self.global_encoder_state = None
        self.global_decoder_state = None

        # --- 增强边缓存（存储直接嵌入对） ---
        self.aug_pos_pairs = []  # [(z_u, z_v), ...]
        self.aug_neg_pairs = []

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

    # ============== 增强机制（直接嵌入对） ==============

    def inject_augmented_positive_edges(self, pair_list):
        """注入增强正边，pair_list: [(z_u, z_v), ...]"""
        self.aug_pos_pairs.extend([(z_u.detach(), z_v.detach()) for z_u, z_v in pair_list])

    def inject_augmented_negative_edges(self, pair_list):
        """注入增强负边，pair_list: [(z_u, z_v), ...]"""
        self.aug_neg_pairs.extend([(z_u.detach(), z_v.detach()) for z_u, z_v in pair_list])

    def train_on_augmented_positives(self):
        """在增强正边上训练 + FedProx"""
        if not self.aug_pos_pairs:
            return 0.0

        self.encoder.train()
        self.decoder.train()
        self.optimizer.zero_grad()

        aug_pred = []
        for z_u, z_v in self.aug_pos_pairs:
            aug_pred.append(self.decoder(z_u.to(self.device), z_v.to(self.device)))
        aug_pred = torch.cat(aug_pred, dim=0)

        labels = torch.ones(aug_pred.size(0), device=self.device)
        task_loss = self.criterion(aug_pred.squeeze(), labels)

        loss = task_loss + (self.mu / 2.0) * self._compute_prox_reg()
        loss.backward()
        self.optimizer.step()

        self.aug_pos_pairs = []  # 用完清空
        return loss.item()

    def train_on_augmented_negatives(self):
        """在增强负边上训练 + FedProx"""
        if not self.aug_neg_pairs:
            return 0.0

        self.encoder.train()
        self.decoder.train()
        self.optimizer.zero_grad()

        aug_pred = []
        for z_u, z_v in self.aug_neg_pairs:
            aug_pred.append(self.decoder(z_u.to(self.device), z_v.to(self.device)))
        aug_pred = torch.cat(aug_pred, dim=0)

        labels = torch.zeros(aug_pred.size(0), device=self.device)
        task_loss = self.criterion(aug_pred.squeeze(), labels)

        loss = task_loss + (self.mu / 2.0) * self._compute_prox_reg()
        loss.backward()
        self.optimizer.step()

        self.aug_neg_pairs = []  # 用完清空
        return loss.item()

    # ===============================================

    def analyze_prediction_errors(self, cluster_labels, use_test=False, top_percent=0.3):
        """分析误判边，用于辅助增强注入"""
        self.encoder.eval()
        self.decoder.eval()

        false_negatives = defaultdict(int)
        false_positives = defaultdict(int)

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

            pos_pred_label = (torch.sigmoid(pos_pred).squeeze() > 0.5).float()
            neg_pred_label = (torch.sigmoid(neg_pred).squeeze() > 0.5).float()

            fn_mask = (pos_pred_label == 0)
            fp_mask = (neg_pred_label == 1)

            fn_edges = pos_edge_index[:, fn_mask]
            fp_edges = neg_edge_index[:, fp_mask]

            for u, v in fn_edges.t().tolist():
                c1, c2 = cluster_labels[u], cluster_labels[v]
                false_negatives[(c1, c2)] += 1

            for u, v in fp_edges.t().tolist():
                c1, c2 = cluster_labels[u], cluster_labels[v]
                false_positives[(c1, c2)] += 1

        def filter_top_percent(dictionary, top_percent):
            items = list(dictionary.items())
            items.sort(key=lambda x: x[1], reverse=True)
            cutoff = max(1, int(len(items) * top_percent))
            return dict(items[:cutoff])

        return (
            filter_top_percent(false_negatives, top_percent),
            filter_top_percent(false_positives, top_percent)
        )

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
