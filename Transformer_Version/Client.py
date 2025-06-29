import torch
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling
from Model.Transformer import EdgeTransformer


class Client:
    def __init__(self, client_id, data, encoder, transformer, decoder, device='cpu', lr=0.005, weight_decay=1e-4):
        self.client_id = client_id
        self.data = data.to(device)
        self.device = device
        self.encoder = encoder.to(device)
        self.transformer = transformer.to(device)
        self.decoder = decoder.to(device)

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

        loss = self.criterion(pred, labels)
        loss.backward()
        self.optimizer.step()

        return loss.item()

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

        return acc, recall, precision, f1

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
