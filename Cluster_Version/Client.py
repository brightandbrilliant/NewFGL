import torch
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling
from collections import defaultdict


class Client:
    def __init__(self, client_id, data, encoder, decoder, device='cpu', lr=0.005, weight_decay=1e-4):
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
        ]).squeeze()

        pred = torch.cat([pos_pred, neg_pred], dim=0).squeeze()

        loss = self.criterion(pred, labels)
        loss.backward()
        self.optimizer.step()

        return loss.item()

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
            ]).squeeze()

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

    def analyze_prediction_errors(self, cluster_labels, use_test=False):
        """
        返回两个字典：
        - false_negatives[(c1, c2)]: 预测为负但实际上为正的边的聚类对频数
        - false_positives[(c1, c2)]: 预测为正但实际上为负的边的聚类对频数
        """
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

            if fn_mask.ndim == 0: fn_mask = fn_mask.unsqueeze(0)
            if fp_mask.ndim == 0: fp_mask = fp_mask.unsqueeze(0)

            fn_edges = pos_edge_index[:, fn_mask]
            fp_edges = neg_edge_index[:, fp_mask]

            for u, v in fn_edges.t().tolist():
                c1, c2 = cluster_labels[u], cluster_labels[v]
                false_negatives[(c1, c2)] += 1

            for u, v in fp_edges.t().tolist():
                c1, c2 = cluster_labels[u], cluster_labels[v]
                false_positives[(c1, c2)] += 1

        return false_negatives, false_positives

    def get_encoder_state(self):
        return self.encoder.state_dict()

    def get_decoder_state(self):
        return self.decoder.state_dict()

    def set_encoder_state(self, state_dict):
        self.encoder.load_state_dict(state_dict)

    def set_decoder_state(self, state_dict):
        self.decoder.load_state_dict(state_dict)
