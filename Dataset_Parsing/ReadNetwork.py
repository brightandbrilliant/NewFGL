import pickle
import os
from typing import Tuple
import networkx
import matplotlib.pyplot as plt

def read_network(datapath: str, graphidx: int) -> networkx.Graph:
    with open(datapath, 'rb') as f:
        data = pickle.load(f)
    assert type(data) == tuple
    graph_ = data[graphidx]
    return graph_


def inspect_graph_data(graph_: networkx.Graph):
    print(f"节点数量: {graph_.number_of_nodes()}")
    print(f"边数量: {graph_.number_of_edges()}")

    print("前 5 个节点:")
    nodes_printed = 0
    for node, attrs in graph_.nodes(data=True):
        print(f"节点: {node}")
        nodes_printed += 1
        if nodes_printed >= 5:
            break

    print("前 5 条边:")
    edges_printed = 0
    for u, v, attrs in graph_.edges(data=True):
        print(f"边: ({u}, {v})")
        edges_printed += 1
        if edges_printed >= 5:
            break

def draw_static_graph(G, title="Graph", node_color='skyblue'):
    plt.figure(figsize=(10, 10))
    pos = networkx.spring_layout(G, seed=42)  # 使用 spring 布局
    networkx.draw(G, pos,
            node_color=node_color,
            with_labels=False,
            node_size=50,
            edge_color='gray',
            alpha=0.7)
    plt.title(title)
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    data_path = '../dataset/dblp/networks'
    g1 = read_network(data_path, 0)
    g2 = read_network(data_path, 1)
    # inspect_graph_data(g1)
    # inspect_graph_data(g2)
    draw_static_graph(g1, title="G1 - Network")
    draw_static_graph(g2, title="G2 - Network")
