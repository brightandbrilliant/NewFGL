import os
import torch
from Client import Client
from Model.GCN import GCN
from Model.ResMLP import ResMLP
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import to_undirected
from Cluster import (
    kmeans_cluster_single,
    compute_anchor_feature_differences,
    build_cluster_cooccurrence_matrix,
    extract_clear_alignments
)
from Parse_Anchors import read_anchors, parse_anchors
from Build import build_positive_edge_dict, build_edge_type_alignment


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

    data = train_data
    val_mask = val_data.edge_label.bool()
    test_mask = test_data.edge_label.bool()
    data.val_pos_edge_index = val_data.edge_label_index[:, val_mask]
    data.val_neg_edge_index = val_data.edge_label_index[:, ~val_mask]
    data.test_pos_edge_index = test_data.edge_label_index[:, test_mask]
    data.test_neg_edge_index = test_data.edge_label_index[:, ~test_mask]

    return train_data


def load_all_clients(pyg_data_paths, encoder_params, decoder_params, training_params, device, nClusters=10, enhance_interval=5):
    clients, all_cluster_labels, raw_data_list, edge_dicts = [], [], [], []

    for client_id, path in enumerate(pyg_data_paths):
        raw_data = torch.load(path)
        raw_data_list.append(raw_data)
        data = split_client_data(raw_data)

        cluster_labels, _ = kmeans_cluster_single(data, n_clusters=nClusters)
        all_cluster_labels.append(cluster_labels)

        edge_dict = build_positive_edge_dict(data, cluster_labels)
        edge_dicts.append(edge_dict)

        encoder = GCN(**encoder_params)
        decoder = ResMLP(input_dim=encoder_params['output_dim'] * 2, **decoder_params)

        client = Client(
            client_id=client_id,
            data=data,
            encoder=encoder,
            decoder=decoder,
            device=device,
            lr=training_params['lr'],
            weight_decay=training_params['weight_decay'],
            enhance_interval=enhance_interval
        )
        clients.append(client)

    return clients, all_cluster_labels, raw_data_list, edge_dicts


def average_state_dicts(state_dicts):
    avg_state = {}
    for key in state_dicts[0].keys():
        avg_state[key] = torch.stack([sd[key] for sd in state_dicts], dim=0).mean(dim=0)
    return avg_state


def extract_augmented_positive_edges(target_fp_types, edge_dict, edge_alignment, top_k=100):
    selected_edges = []
    for (c1, c2) in target_fp_types:
        aligned_targets = edge_alignment.get((c1, c2), [])
        for (c1_p, c2_p), weight in aligned_targets:
            candidate_edges = edge_dict.get((c1_p, c2_p), [])
            selected_edges.extend(candidate_edges[:top_k])
    return selected_edges


def evaluate_all_clients(clients, cluster_labels, use_test=False):
    metrics = []
    for i, client in enumerate(clients):
        acc, recall, precision, f1 = client.evaluate(use_test=use_test)
        metrics.append((acc, recall, precision, f1))
        print(f"Client {client.client_id}: Acc={acc:.4f}, Recall={recall:.4f}, Prec={precision:.4f}, F1={f1:.4f}")
    avg = torch.tensor(metrics).mean(dim=0).tolist()
    print(f"\n===> Average: Acc={avg[0]:.4f}, Recall={avg[1]:.4f}, Prec={avg[2]:.4f}, F1={avg[3]:.4f}")
    return avg


if __name__ == "__main__":
    data_dir = "../Parsed_dataset/dblp"
    anchor_path = "../dataset/dblp/anchors.txt"
    pyg_data_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".pt")])

    encoder_params = {
        'input_dim': torch.load(pyg_data_files[0]).x.shape[1],
        'hidden_dim': 128,
        'output_dim': 64,
        'num_layers': 3,
        'dropout': 0.5
    }
    decoder_params = {'hidden_dim': 128, 'num_layers': 8, 'dropout': 0.3}
    training_params = {'lr': 0.001, 'weight_decay': 1e-4, 'local_epochs': 5}

    num_rounds = 600
    augment_start_round = 300
    top_fp_percent = 0.3
    enhance_interval = 5
    top_k_per_type = 100
    nClusters = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Flag to choose between using your method or full-positive-edge injection method
    use_all_positive_edges = False  # Set this flag to True if you want to inject all positive edges from the other client

    clients, cluster_labels, raw_data_list, edge_dicts = load_all_clients(
        pyg_data_files, encoder_params, decoder_params, training_params, device, nClusters, enhance_interval
    )

    anchor_raw = read_anchors(anchor_path)
    anchor_pairs = parse_anchors(anchor_raw, point=9086)
    results = compute_anchor_feature_differences(raw_data_list[0], raw_data_list[1], anchor_pairs)
    co_matrix = build_cluster_cooccurrence_matrix(cluster_labels[0], cluster_labels[1], results, nClusters, top_percent=0.75)
    alignment1 = extract_clear_alignments(co_matrix, min_ratio=0.25, min_count=30, mode=1)
    alignment2 = extract_clear_alignments(co_matrix, min_ratio=0.25, min_count=30, mode=2)
    edge_alignment1 = build_edge_type_alignment(alignment1, nClusters)
    edge_alignment2 = build_edge_type_alignment(alignment2, nClusters)

    best_f1 = -1
    best_encoder_state = None
    best_decoder_states = None

    print("\n================ Federated Training Start ================")
    for rnd in range(1, num_rounds + 1):
        print(f"\n--- Round {rnd} ---")

        if rnd == augment_start_round:
            print("\n===> Injecting hard negatives and augmented positives")
            z_others = [client.encoder(client.data.x, client.data.edge_index).detach() for client in clients]

            for i, client in enumerate(clients):
                fn, fp = client.analyze_prediction_errors(cluster_labels[i], use_test=False, top_percent=top_fp_percent)
                client.inject_hard_negatives(fp, cluster_labels[i], max_per_pair=300)

                # Inject all positive edges or selective augmented positive edges based on the flag
                if use_all_positive_edges:
                    # Inject all positive edges from the other client
                    j = 1 - i  # Assuming two clients, if i=0, j=1 and vice versa
                    client.inject_all_positive_edges_from_other_client(clients[j])
                else:
                    # Inject augmented positive edges (your method)
                    j = 1 - i
                    edge_list = extract_augmented_positive_edges(fp, edge_dicts[j], edge_alignment1 if i == 0 else edge_alignment2, top_k=top_k_per_type)
                    client.inject_augmented_positive_edges(edge_list, z_others[j])

        for client in clients:
            for _ in range(training_params['local_epochs']):
                client.train()
            if rnd >= augment_start_round and rnd % enhance_interval == 0:
                client.train_on_hard_negatives()
                client.train_on_augmented_positives()

        encoder_states = [client.get_encoder_state() for client in clients]
        decoder_states = [client.get_decoder_state() for client in clients]
        global_encoder_state = average_state_dicts(encoder_states)

        for client in clients:
            client.set_encoder_state(global_encoder_state)

        avg_acc, avg_recall, avg_prec, avg_f1 = evaluate_all_clients(clients, cluster_labels, use_test=False)

        if avg_f1 > best_f1:
            best_f1 = avg_f1
            best_encoder_state = global_encoder_state
            best_decoder_states = decoder_states
            print("===> New best model saved")

    print("\n================ Federated Training Finished ================")
    for i, client in enumerate(clients):
        client.set_encoder_state(best_encoder_state)
        client.set_decoder_state(best_decoder_states[i])

    print("\n================ Final Evaluation ================")
    evaluate_all_clients(clients, cluster_labels, use_test=True)
