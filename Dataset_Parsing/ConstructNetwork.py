from ReadNetwork import read_network
from ReadUserFeatures import read_user_features
import os
import pickle
import numpy as np
import torch
from torch_geometric.data import Data
import networkx as nx
from typing import Dict, Tuple, List


def build_local_pyg_graphs(
    save_path_prefix: str,
    graphs: Tuple[nx.Graph, ...],
    user_features: Dict[int, np.ndarray],
    node_id_ranges: List[Tuple[int, int]]
):
    """
    为多个 NetworkX 图构建 PyG 对象，使用局部编号（每张图独立编号）。
    图2节点将统一减去偏移量（如10000）以形成自己的编号空间。

    Args:
        save_path_prefix (str): 保存路径前缀。
        graphs (Tuple[nx.Graph, ...]): 每个图为 NetworkX 对象。
        user_features (Dict[int, np.ndarray]): 原始全局节点特征。
        node_id_ranges (List[Tuple[int, int]]): 每个图的原始编号范围，如 [(0, 9999), (10000,19999)]。
    """
    for i, (g, (start_id, end_id)) in enumerate(zip(graphs, node_id_ranges)):
        print(f"\n--- 处理图 Graph_{i+1} ---")
        g.name = f"Graph_{i + 1}"

        offset = start_id  # 局部编号偏移
        local_id_map = {}  # 原始id -> 局部id 映射
        local_nodes = list(g.nodes())

        # 创建映射
        for local_idx, original_id in enumerate(sorted(local_nodes)):
            local_id_map[original_id] = original_id - offset

        # 提取特征维度
        sample_feat = next(iter(user_features.values()))
        feat_dim = sample_feat.shape[0]

        # 构建局部特征矩阵
        num_nodes = len(local_nodes)
        local_features = np.zeros((num_nodes, feat_dim), dtype=np.float32)
        for original_id in local_nodes:
            if original_id in user_features:
                local_features[local_id_map[original_id]] = user_features[original_id]
            else:
                print(f"警告: 节点 {original_id} 缺少特征，使用全0填充")

        x = torch.tensor(local_features, dtype=torch.float)

        # 构建局部 edge_index
        edge_list = []
        for u, v in g.edges():
            if u in local_id_map and v in local_id_map:
                u_local = local_id_map[u]
                v_local = local_id_map[v]
                edge_list.append([u_local, v_local])
                edge_list.append([v_local, u_local])  # 无向图加反向边
            else:
                print(f"边 ({u},{v}) 有节点不在映射中，跳过")

        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous() if edge_list else torch.empty((2, 0), dtype=torch.long)

        data = Data(x=x, edge_index=edge_index)

        # 保存数据
        save_path = f"{save_path_prefix}_{i + 1}.pt"
        torch.save(data, save_path)
        print(f"保存至 {save_path}，节点数: {data.num_nodes}，边数: {data.num_edges}")


if __name__ == "__main__":
    data_path_prefix = "../Parsed_dataset/wd/wd"
    gdata_path = '../dataset/wd/networks'
    g1 = read_network(gdata_path, 0)
    g2 = read_network(gdata_path, 1)
    graphs = (g1, g2)
    fdatapath = "../dataset/wd/user_features.pkl"
    user_features = read_user_features(fdatapath)
    node_id_ranges = [(0, 9713), (9714, 19239)]
    build_local_pyg_graphs( data_path_prefix, graphs, user_features, node_id_ranges)




