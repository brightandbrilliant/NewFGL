import os
import torch
from Client import Client
from Model.GCN import GCN
from Model.ResMLP import ResMLP
# from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import to_undirected
from Cluster import kmeans_cluster_single, compute_anchor_feature_differences, build_cluster_cooccurrence_matrix, extract_clear_alignments
from Parse_Anchors import read_anchors, parse_anchors
from collections import defaultdict


def build_edge_type_alignment(alignment, nClusters):
    """
    保留边的方向，构建从源边类型 (i,j) 到目标边类型 (i',j') 的映射。
    返回: dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]]
    """
    edge_mapping = {}

    for i in range(nClusters):
        aligned_i = alignment.get(i, [])  # [(j, score)]
        for j in range(nClusters):
            aligned_j = alignment.get(j, [])  # 保留顺序，不做排序
            key = (i, j)  # 有向边类型

            mapped = []
            for (i_, si) in aligned_i:
                for (j_, sj) in aligned_j:
                    target_key = (i_, j_)
                    weight = si * sj
                    mapped.append((target_key, weight))

            # 合并相同 target_key 的权重
            merged = {}
            for k, w in mapped:
                merged[k] = merged.get(k, 0) + w
            mapped_list = [(k, w) for k, w in merged.items()]
            edge_mapping[key] = mapped_list

    return edge_mapping



def build_positive_edge_dict(data, cluster_labels):
    edge_dict = defaultdict(list)
    edge_index = data.edge_index
    for u, v in edge_index.t().tolist():
        c1, c2 = cluster_labels[u], cluster_labels[v]
        key = (c1, c2)
        edge_dict[key].append((u, v))
    return edge_dict






