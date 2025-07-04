import os
import torch
from Client import Client
from Model.GCN import GCN
from Model.ResMLP import ResMLP
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import to_undirected
from Cluster import kmeans_cluster_single


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


def load_all_clients(pyg_data_paths, encoder_params, decoder_params, training_params, device, nClusters=10):
    clients = []
    all_cluster_labels = []
    for client_id, path in enumerate(pyg_data_paths):
        raw_data = torch.load(path)
        data = split_client_data(raw_data)

        cluster_labels, _ = kmeans_cluster_single(data, n_clusters=nClusters)
        all_cluster_labels.append(cluster_labels)

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
            device=device,
            lr=training_params['lr'],
            weight_decay=training_params['weight_decay']
        )
        clients.append(client)
    return clients, all_cluster_labels


def average_state_dicts(state_dicts):
    avg_state = {}
    for key in state_dicts[0].keys():
        avg_state[key] = torch.stack([sd[key] for sd in state_dicts], dim=0).mean(dim=0)
    return avg_state


def evaluate_all_clients(clients, cluster_labels, use_test=False):
    metrics = []
    for i, client in enumerate(clients):
        acc, recall, precision, f1 = client.evaluate(use_test=use_test)
        metrics.append((acc, recall, precision, f1))
        print(f"Client {client.client_id}: Acc={acc:.4f}, Recall={recall:.4f}, "
              f"Prec={precision:.4f}, F1={f1:.4f}")

        # 分析误判聚类
        fn, fp = client.analyze_prediction_errors(cluster_labels[i], use_test=use_test)
        print(f"  False Negative Cluster Pairs: {dict(fn)}")
        print(f"  False Positive Cluster Pairs: {dict(fp)}")

    avg_metrics = torch.tensor(metrics).mean(dim=0).tolist()
    print(f"\n===> Average Metrics: Acc={avg_metrics[0]:.4f}, Recall={avg_metrics[1]:.4f}, "
          f"Prec={avg_metrics[2]:.4f}, F1={avg_metrics[3]:.4f}")
    return avg_metrics


if __name__ == "__main__":
    # 1. 配置路径与参数
    data_dir = "../Parsed_dataset/dblp"
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

    num_rounds = 600
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    nClusters = 10

    # 2. 初始化客户端
    clients, cluster_labels = load_all_clients(pyg_data_files, encoder_params, decoder_params, training_params, device, nClusters)

    best_f1 = -1
    best_encoder_state = None
    best_decoder_states = None

    print("\n================ Federated Training Start ================\n")
    for rnd in range(1, num_rounds + 1):
        print(f"\n--- Round {rnd} ---")

        # 3. 每个客户端本地训练
        for client in clients:
            for _ in range(training_params['local_epochs']):
                loss = client.train()

        # 4. FedAvg 聚合参数
        encoder_states = [client.get_encoder_state() for client in clients]
        decoder_states = [client.get_decoder_state() for client in clients]

        global_encoder_state = average_state_dicts(encoder_states)

        # 5. 同步参数
        for client in clients:
            client.set_encoder_state(global_encoder_state)

        # 6. 联邦评估
        avg_acc, avg_recall, avg_prec, avg_f1 = evaluate_all_clients(clients, cluster_labels, use_test=False)

        if avg_f1 > best_f1:
            best_f1 = avg_f1
            best_encoder_state = global_encoder_state
            best_decoder_states = decoder_states
            print("===> New best global model saved.")

    print("\n================ Federated Training Finished ================\n")

    # 7. 最终模型评估
    for i, client in enumerate(clients):
        client.set_encoder_state(best_encoder_state)
        client.set_decoder_state(best_decoder_states[i])

    print("\n================ Final Evaluation ================")
    evaluate_all_clients(clients, cluster_labels, use_test=True)
