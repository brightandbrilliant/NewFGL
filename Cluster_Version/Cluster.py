import os
import torch
from torch_geometric.data import Data
from Parse_Anchors import read_anchors, parse_anchors
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

def load_pyg_data(file_path: str) -> Data:
    data = torch.load(file_path)
    if not isinstance(data, Data):
        raise ValueError(f"文件 {file_path} 中的对象不是 torch_geometric.data.Data 类型。")
    return data

def load_anchors(file_path: str):
    anchor_list = read_anchors(file_path)
    point = 9086
    anchor_list = parse_anchors(anchor_list, point)
    return anchor_list


def compute_diff(f1, f2, mode='euclidean'):
    if mode == 'euclidean':
        return torch.norm(f1 - f2, p=2).item()
    elif mode == 'cosine':
        return 1 - F.cosine_similarity(f1.unsqueeze(0), f2.unsqueeze(0)).item()
    elif mode == 'dot':
        return torch.dot(f1, f2).item()
    else:
        raise ValueError(f"Unknown mode: {mode}")


def compute_anchor_feature_differences(data1, data2, anchor_pairs):
    results = []
    for pair in anchor_pairs:
        idx1, idx2 = pair[0], pair[1]
        f1, f2 = data1.x[idx1], data2.x[idx2]
        diff = compute_diff(f1, f2, 'dot')
        results.append([idx1, idx2, diff])
    return results




def plot_anchor_diff_distribution(results, bins=100, show_kde=True, title="Anchor Feature Differences"):
    # 提取差异度列表
    diffs = [item[2] for item in results]

    plt.figure(figsize=(8, 5))
    sns.histplot(diffs, bins=bins, kde=show_kde, color='skyblue', edgecolor='black')

    plt.title(title)
    plt.xlabel("Feature Difference")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def assign_weights_by_threshold(results, thresholds=[0.1, 0.35], weights=[1.0, 0.5, 0.1]):
    """
    根据差异度阈值将每对锚点划分类别并分配权重。

    Args:
        results: 形如 [[idx1, idx2, diff], ...]
        thresholds: 差异度划分的阈值列表，长度 = 类别数 - 1
        weights: 每个类别对应的权重（应比类别数目相同）

    Returns:
        weighted_anchors: [[idx1, idx2, diff, weight], ...]
    """
    weighted_anchors = []
    for idx1, idx2, diff in results:
        if diff <= thresholds[0]:
            weight = weights[0]
        elif diff <= thresholds[1]:
            weight = weights[1]
        else:
            weight = weights[2]
        weighted_anchors.append([idx1, idx2, diff, weight])
    return weighted_anchors


if __name__ == "__main__":
    pyg_path1 = "../Parsed_dataset/dblp/dblp_1.pt"
    pyg_path2 = "../Parsed_dataset/dblp/dblp_2.pt"
    data1, data2 = load_pyg_data(pyg_path1), load_pyg_data(pyg_path2)
    anchor_path = "../dataset/dblp/anchors.txt"
    anchor_pairs = load_anchors(anchor_path)
    results = compute_anchor_feature_differences(data1, data2, anchor_pairs)
    print(results)
    print(len(results))
    # plot_anchor_diff_distribution(results)

