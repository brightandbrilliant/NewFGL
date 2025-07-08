import torch
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling
from collections import defaultdict
import random


class Client:
    def __init__(self, client_id, data, encoder, decoder, device='cpu', lr=0.005, weight_decay=1e-4, enhance_interval=100):
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
        self.hard_neg_edges = None
        self.augmented_pos_embeddings = None
        self.enhance_interval = enhance_interval
        self.all_pos_edges_from_other_client = None  # 新增：对方的所有正边

    def train(self):
        self.encoder.train()
        self.decoder.train()
        self.optimizer.zero_grad()

        pos_edge_index = self.data.edge_index

        # 通过注入对方所有正边
        if self.all_pos_edges_from_other_client is not None:
            pos_edge_index = torch.cat([pos_edge_index, self.all_pos_edges_from_other_client], dim=1)

        neg_edge_index = negative_sampling(
            edge_index=pos_edge_index,
            num_nodes=self.data.num_nodes,
            num_neg_samples=pos_edge_index.size(1)
        )

        if self.hard_neg_edges is not None:
            neg_edge_index = torch.cat([neg_edge_index, self.hard_neg_edges.to(self.device)], dim=1)

        z = self.encoder(self.data.x, self.data.edge_index)
        pos_pred = self.decoder(z[pos_edge_index[0]], z[pos_edge_index[1]])
        neg_pred = self.decoder(z[neg_edge_index[0]], z[neg_edge_index[1]])

        labels = torch.cat([
            torch.ones(pos_pred.size(0), device=self.device),
            torch.zeros(neg_pred.size(0), device=self.device)
        ])
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        loss = self.criterion(pred.squeeze(), labels)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    # 新增：注入对方的所有正边
    def inject_all_positive_edges_from_other_client(self, other_client):
        """
        注入对方客户端的所有正边
        :param other_client: 另一个客户端
        """
        if other_client.data.edge_index is not None:
            self.all_pos_edges_from_other_client = other_client.data.edge_index
        else:
            self.all_pos_edges_from_other_client = None

    def train_on_hard_negatives(self):
        if self.hard_neg_edges is None:
            return 0.0

        self.encoder.train()
        self.decoder.train()
        self.optimizer.zero_grad()

        z = self.encoder(self.data.x, self.data.edge_index)
        pos_pred = self.decoder(z[self.data.edge_index[0]], z[self.data.edge_index[1]])
        neg_pred = self.decoder(z[self.hard_neg_edges[0]], z[self.hard_neg_edges[1]])

        labels = torch.cat([
            torch.ones(pos_pred.size(0), device=self.device),
            torch.zeros(neg_pred.size(0), device=self.device)
        ])
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        loss = self.criterion(pred.squeeze(), labels.squeeze())
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train_on_augmented_positives(self):
        if self.augmented_pos_embeddings is None:
            return 0.0

        self.encoder.train()
        self.decoder.train()
        self.optimizer.zero_grad()

        z = self.encoder(self.data.x, self.data.edge_index)
        pos_pred = self.decoder(z[self.data.edge_index[0]], z[self.data.edge_index[1]])

        z_u_aug, z_v_aug = zip(*self.augmented_pos_embeddings)
        z_u_aug = torch.stack(z_u_aug).to(self.device)
        z_v_aug = torch.stack(z_v_aug).to(self.device)
        pos_pred_aug = self.decoder(z_u_aug, z_v_aug)

        labels = torch.ones(pos_pred.size(0) + pos_pred_aug.size(0), device=self.device)
        pred = torch.cat([pos_pred, pos_pred_aug], dim=0)

        loss = self.criterion(pred.squeeze(), labels)
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

    def analyze_prediction_errors(self, cluster_labels, use_test=False, top_percent=0.3):
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

        filtered_fn = filter_top_percent(false_negatives, top_percent)
        filtered_fp = filter_top_percent(false_positives, top_percent)

        return filtered_fn, filtered_fp

    def inject_hard_negatives(self, target_pairs, cluster_labels, max_per_pair=300):
        self.encoder.eval()
        z = self.encoder(self.data.x, self.data.edge_index)

        label_to_nodes = defaultdict(list)
        for node, label in enumerate(cluster_labels):
            label_to_nodes[label].append(node)

        existing_edges = set((u.item(), v.item()) for u, v in self.data.edge_index.t())
        hard_candidates = []

        for (c1, c2) in target_pairs:
            nodes1 = label_to_nodes.get(c1, [])
            nodes2 = label_to_nodes.get(c2, [])

            sampled = []
            for u in nodes1:
                for v in random.sample(nodes2, min(len(nodes2), 20)):
                    if u != v and (u, v) not in existing_edges and (v, u) not in existing_edges:
                        sampled.append((u, v))
                    if len(sampled) >= max_per_pair:
                        break
                if len(sampled) >= max_per_pair:
                    break

            for u, v in sampled:
                score = torch.sigmoid(self.decoder(z[u], z[v])).item()
                hard_candidates.append(((u, v), score))

        if not hard_candidates:
            self.hard_neg_edges = None
            return

        hard_candidates.sort(key=lambda x: x[1], reverse=True)
        top_k = min(max_per_pair * len(target_pairs), len(hard_candidates))
        selected = hard_candidates[:top_k]
        edge_index = torch.tensor([list(e) for e, _ in selected], dtype=torch.long).t().contiguous()
        self.hard_neg_edges = edge_index.to(self.device)

    def inject_augmented_positive_edges(self, edge_list, other_embeddings):
        if edge_list:
            self.augmented_pos_embeddings = [
                (other_embeddings[u].detach(), other_embeddings[v].detach())
                for u, v in edge_list
            ]
        else:
            self.augmented_pos_embeddings = None

    def clear_augmented_edges(self):
        self.augmented_pos_embeddings = None

    def get_encoder_state(self):
        return self.encoder.state_dict()

    def get_decoder_state(self):
        return self.decoder.state_dict()

    def set_encoder_state(self, state_dict):
        self.encoder.load_state_dict(state_dict)

    def set_decoder_state(self, state_dict):
        self.decoder.load_state_dict(state_dict)
