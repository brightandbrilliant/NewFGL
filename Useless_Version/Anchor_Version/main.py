import os
import torch
from Client import Client
from Model.GCN import GCN
from Model.ResMLP import ResMLP
from torch_geometric.data import Data
from Parse_Anchors import read_anchors, parse_anchors
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import to_undirected

def split_client_data(data, val_ratio=0.1, test_ratio=0.1, device='cpu'):
    data = data.to(device)
    data.edge_index = to_undirected(data.edge_index, num_nodes=data.num_nodes)

    transform = RandomLinkSplit(
        num_val=val_ratio,
        num_test=test_ratio,
        is_undirected=True,
        neg_sampling_ratio=1.0
    )
    train_data, val_data, test_data = transform(data)

    val_mask = val_data.edge_label.bool()
    test_mask = test_data.edge_label.bool()
    train_data.val_pos_edge_index = val_data.edge_label_index[:, val_mask]
    train_data.val_neg_edge_index = val_data.edge_label_index[:, ~val_mask]
    train_data.test_pos_edge_index = test_data.edge_label_index[:, test_mask]
    train_data.test_neg_edge_index = test_data.edge_label_index[:, ~test_mask]

    return train_data


def load_all_clients(pyg_data_paths, anchor_list, encoder_params, decoder_params,
                     training_params, device, contrastive_weight):
    clients = []
    for client_id, path in enumerate(pyg_data_paths):
        raw_data = torch.load(path)
        data = split_client_data(raw_data)

        encoder = GCN(
            input_dim=encoder_params['input_dim'],
            hidden_dim=encoder_params['hidden_dim'],
            output_dim=encoder_params['output_dim'],
            num_layers=encoder_params['num_layers'],
            dropout=encoder_params['dropout']
        )

        decoder = ResMLP(
            input_dim=encoder_params['output_dim'] * 2,
            hidden_dim=decoder_params['hidden_dim'],
            num_layers=decoder_params['num_layers'],
            dropout=decoder_params['dropout']
        )

        client = Client(
            client_id=client_id,
            data=data,
            encoder=encoder,
            decoder=decoder,
            anchor_list=anchor_list,
            contrastive_weight=contrastive_weight,
            device=device,
            lr=training_params['lr'],
            weight_decay=training_params['weight_decay']
        )
        clients.append(client)
    return clients


def average_state_dicts(state_dicts):
    avg_state = {}
    for key in state_dicts[0].keys():
        avg_state[key] = torch.stack([sd[key] for sd in state_dicts], dim=0).mean(dim=0)
    return avg_state


def evaluate_all_clients(clients, use_test=False):
    metrics = []
    zs = []
    for client in clients:
        acc, recall, precision, f1, z = client.evaluate(use_test)
        metrics.append((acc, recall, precision, f1))
        zs.append(z)
        print(f"Client {client.client_id}: Acc={acc:.4f}, Recall={recall:.4f}, "
              f"Prec={precision:.4f}, F1={f1:.4f}")
    avg_metrics = torch.tensor(metrics).mean(dim=0).tolist()
    print(f"\n===> Average Metrics: Acc={avg_metrics[0]:.4f}, Recall={avg_metrics[1]:.4f}, "
          f"Prec={avg_metrics[2]:.4f}, F1={avg_metrics[3]:.4f}")
    return avg_metrics, zs


if __name__ == "__main__":
    # 1. 配置路径与参数
    data_dir = "../../Parsed_dataset/dblp"
    anchor_path = "../../dataset/dblp/anchors.txt"
    pyg_data_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".pt")])

    encoder_params = {
        'input_dim': torch.load(pyg_data_files[0]).x.shape[1],
        'hidden_dim': 128,
        'output_dim': 64,
        'num_layers': 3,
        'dropout': 0.5
    }

    decoder_params = {
        'hidden_dim': 128,
        'num_layers': 8,
        'dropout': 0.3
    }

    training_params = {
        'lr': 0.001,
        'weight_decay': 1e-4,
        'local_epochs': 5
    }

    contrastive_weight = 1.0
    num_rounds = 600
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    anchor_list = read_anchors(anchor_path)
    point = 9086
    anchor_list = parse_anchors(anchor_list, point)
    clients = load_all_clients(pyg_data_files, anchor_list, encoder_params,
                               decoder_params, training_params, device, contrastive_weight)

    best_f1 = -1
    best_encoder_state = None
    best_decoder_states = None

    print("\n================ Federated Training Start ================\n")
    for rnd in range(1, num_rounds + 1):
        print(f"\n--- Round {rnd} ---")

        # 每轮前：各客户端注入 peer_z（对比目标嵌入）
        with torch.no_grad():
            zs = [client.encoder(client.data.x.to(device), client.data.edge_index.to(device)) for client in clients]
        clients[0].peer_z = zs[1].detach()
        clients[1].peer_z = zs[0].detach()

        for client in clients:
            for _ in range(training_params['local_epochs']):
                loss = client.train()

        encoder_states = [client.get_encoder_state() for client in clients]
        decoder_states = [client.get_decoder_state() for client in clients]

        global_encoder_state = average_state_dicts(encoder_states)

        for client in clients:
            client.set_encoder_state(global_encoder_state)

        metrics, zs_ = evaluate_all_clients(clients, False)

        if metrics[3] > best_f1:
            best_f1 = metrics[3]
            best_encoder_state = global_encoder_state
            best_decoder_states = decoder_states
            print("===> New best global model saved.")

    print("\n================ Federated Training Finished ================\n")

    for i, client in enumerate(clients):
        client.set_encoder_state(best_encoder_state)
        client.set_decoder_state(best_decoder_states[i])

    print("\n================ Final Evaluation ================")
    evaluate_all_clients(clients, True)
