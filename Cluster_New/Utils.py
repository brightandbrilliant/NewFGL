from collections import defaultdict, deque
import numpy as np
import torch

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


def has_excessive_oscillations(loss_window) -> bool:
    if len(loss_window) < 5:
        return True  # 数据太少，保守地认为仍处于下降

    delta = np.diff(loss_window)
    mean, std = np.mean(delta), np.std(delta)
    # 统计大于 2 倍标准差的震荡点数量
    num_spikes = np.sum(np.abs(delta - mean) > 2 * std)

    # 若大幅震荡超过20%则判定为仍在剧烈变化期
    return num_spikes >= max(1, len(delta) // 5)


def judge_loss_window_poly(loss_window, second_deriv_window, deg=4):
    """
    用多项式拟合loss窗口并判断是否已脱离快速下降期。
    参数:
        loss_window: 长度为100的 loss 序列（list 或 deque）
        deg: 多项式拟合的阶数，默认5阶
    返回:
        bool：True 表示已趋于平稳；False 表示仍在快速下降
    """
    if isinstance(loss_window, torch.Tensor):
        y = loss_window.cpu().numpy()
    else:
        y = np.array(loss_window)

    x = np.arange(len(y))

    # 多项式拟合
    coefs = np.polyfit(x, y, deg=deg)
    poly = np.poly1d(coefs)

    # 计算二阶导（对应二次导数系数 * 2 * 1）
    second_deriv = np.polyder(poly, m=2)
    second_values = second_deriv(x)

    second_value = np.mean(second_values)

    second_deriv_window.append(second_value)

    # 计算差分序列
    diffs = np.diff(second_deriv_window[-5:])  # 最近 5 个导数的变化趋势
    curvature_flattened = second_deriv_window[-1] > -1e-5

    # 如果二阶导的变化趋势是持续向上（意味着凹性持续减弱）
    flag = all(d > 0 for d in diffs) and curvature_flattened

    return flag, second_deriv_window


# 返回 True 则进入增强期，返回 False 则不进入增强期
def judge_loss_window(loss_window: deque, second_deriv_window: deque, deg: int):
    if len(loss_window) < 100:
        return False, second_deriv_window
    if has_excessive_oscillations(loss_window) is True:
        return False, second_deriv_window

    flag, second_deriv_window_ = judge_loss_window_poly(loss_window, second_deriv_window, deg)
    if len(second_deriv_window_) < 5:
        return False, second_deriv_window_
    else:
        return flag, second_deriv_window_

