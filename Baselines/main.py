import torch
from torch_geometric.utils import negative_sampling, to_undirected  # <--- Import to_undirected here
from torch_geometric.transforms import RandomLinkSplit
from typing import Dict
from Evaluate.Evaluate import evaluate_accuracy_recall_f1

# --- 从 Model 目录导入你的模型组件 ---
from Baselines.Model.GCN import GCN
from Baselines.Model.ResMLP import ResMLP


def train_link_prediction_model(
        pyg_data_path: str,
        encoder_params: Dict,
        decoder_params: Dict,
        training_params: Dict
):
    """
    链接预测模型的端到端训练框架。

    Args:
        pyg_data_path (str): 预处理好的 PyG Data 对象的路径 (例如 'my_fused_social_graph_1.pt')。
        encoder_params (Dict): 编码器 (ResidualGCN) 的初始化参数字典，
                                包含 'input_dim', 'hidden_dim', 'output_dim', 'num_layers', 'dropout'。
        decoder_params (Dict): 解码器 (ResMLPDecoder) 的初始化参数字典，
                                包含 'hidden_dim', 'num_layers', 'dropout'。
        training_params (Dict): 训练相关的参数字典，
                                包含 'val_ratio', 'test_ratio', 'lr', 'epochs', 'log_every_epochs'。
    """

    # 1. 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"当前使用的设备: {device}")

    # 2. 数据加载与划分
    print(f"\n--- 1. 加载并划分数据: {pyg_data_path} ---")

    # 首先加载原始的完整图数据
    original_data = torch.load(pyg_data_path)
    original_data = original_data.to(device)  # 将原始数据移到设备

    # --- NEW: Explicitly make the graph undirected before RandomLinkSplit ---
    # This ensures that all edges (including those that will be split) have inverse edges.
    # edge_attr is None for your current setup, so it's fine.
    original_data.edge_index = to_undirected(
        original_data.edge_index, num_nodes=original_data.num_nodes
    )

    print(f"原始图已转换为无向图。")

    # 实例化 RandomLinkSplit 变换
    transform = RandomLinkSplit(
        num_val=training_params['val_ratio'],
        num_test=training_params['test_ratio'],
        is_undirected=True,  # 仍然保持此项，它控制了val/test样本的生成逻辑
        neg_sampling_ratio=1.0
    )

    # 应用变换
    train_data, val_data, test_data = transform(original_data)

    data = train_data
    val_mask = val_data.edge_label.bool()
    test_mask = test_data.edge_label.bool()
    data.val_pos_edge_index = val_data.edge_label_index[:, val_mask]
    data.val_neg_edge_index = val_data.edge_label_index[:, ~val_mask]
    data.test_pos_edge_index = test_data.edge_label_index[:, test_mask]
    data.test_neg_edge_index = test_data.edge_label_index[:, ~test_mask]

    if data.edge_index is None or data.edge_index.numel() == 0:
        raise ValueError(f"Error: Training graph (data.edge_index) is empty after RandomLinkSplit. "
                         f"Original graph had {original_data.edge_index.size(1) // 2} edges (before split). "
                         f"Please check your graph data or split ratios (val_ratio, test_ratio).")

    print(f"总节点数: {data.num_nodes}")
    print(f"GCN 消息传递的边对数 (data.edge_index): {data.edge_index.size(1)}")
    # RandomLinkSplit 的 train_data.edge_label_index 已经包含了正负样本

    # 3. 模型初始化
    print("\n--- 2. 初始化编码器和解码器 ---")
    encoder = GCN(input_dim=data.x.shape[1],  # 从数据中获取输入特征维度
                     hidden_dim=encoder_params['hidden_dim'],
                     output_dim=encoder_params['output_dim'],
                     num_layers=encoder_params['num_layers'],
                     dropout=encoder_params['dropout']).to(device)

    decoder = ResMLP(input_dim=encoder_params['output_dim'] * 2,  # 解码器输入维度是编码器输出的两倍
                     hidden_dim=decoder_params['hidden_dim'],
                     num_layers=decoder_params['num_layers'],
                     dropout=decoder_params['dropout']).to(device)

    # 4. 损失函数与优化器
    print("\n--- 3. 配置损失函数和优化器 ---")
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),
                                 lr=training_params['lr'], weight_decay=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss()
    print(f"损失函数: {criterion}")

    # 5. 训练循环
    print("\n--- 4. 开始训练 ---")
    best_val_f1 = -1.0  # 记录最佳验证AUC
    best_encoder_state = None
    best_decoder_state = None

    for epoch in range(1, training_params['epochs'] + 1):
        encoder.train()
        decoder.train()
        optimizer.zero_grad()

        # GCN 编码器生成节点嵌入，使用 train_data.edge_index
        z = encoder(data.x, data.edge_index)

        # 获取训练集中的原始正样本边
        train_pos_edge_index_for_loss = data.edge_label_index[:, data.edge_label.bool()]

        # 动态负采样 for 训练集
        # 使用 GCN 实际看到的训练图的边来采样负样本
        neg_train_edge_index = negative_sampling(
            edge_index=data.edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=train_pos_edge_index_for_loss.size(1)
        ).to(device)

        pos_train_pred = decoder(z[train_pos_edge_index_for_loss[0]], z[train_pos_edge_index_for_loss[1]])
        neg_train_pred = decoder(z[neg_train_edge_index[0]], z[neg_train_edge_index[1]])

        labels_train = torch.cat([torch.ones(pos_train_pred.size(0)),
                                  torch.zeros(neg_train_pred.size(0))], dim=0).to(device)

        loss = criterion(torch.cat([pos_train_pred, neg_train_pred], dim=0).squeeze(),
                         labels_train.squeeze())

        loss.backward()
        optimizer.step()

        if epoch % training_params['log_every_epochs'] == 0 or epoch == training_params['epochs']:
            val_acc, val_recall, val_pre, val_f1 = evaluate_accuracy_recall_f1(
                encoder, decoder,
                data,
                data.val_pos_edge_index,
                data.val_neg_edge_index,
                device
            )
            print(
                f"Epoch: {epoch:03d}, Loss: {loss:.4f}, "
                f"Val Accuracy: {val_acc:.4f}, Recall: {val_recall:.4f}, precision: {val_pre:.4f}, "
                f"F1: {val_f1:.4f}")

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_encoder_state = encoder.state_dict()
                best_decoder_state = decoder.state_dict()
                print(f"  -> 发现新的最佳验证F1: {best_val_f1:.4f}，模型已保存。")

    print("\n--- 训练完成 ---")

    # 6. 最终测试
    print("\n--- 5. 载入最佳模型进行最终测试 ---")

    encoder.load_state_dict(best_encoder_state)
    decoder.load_state_dict(best_decoder_state)
    test_acc, test_recall, test_pre, test_f1 = evaluate_accuracy_recall_f1(
        encoder, decoder,
        data,
        data.test_pos_edge_index,
        data.test_neg_edge_index,
        device
    )
    print(f"最终测试 Accuracy: {test_acc:.4f}, Recall: {test_recall:.4f},"
          f"  precision: {test_pre:.4f}, F1-score: {test_f1:.4f}")

    return encoder, decoder, test_f1


if __name__ == "__main__":
    pyg_data_file = '../Parsed_dataset/dblp/dblp_1.pt'  # 你的实际数据文件路径

    gcn_output_dim = 64

    temp_data_for_dim = torch.load(pyg_data_file)
    encoder_input_dim_placeholder = temp_data_for_dim.x.shape[1]
    print(f"加载现有数据：{pyg_data_file}，节点特征维度：{encoder_input_dim_placeholder}")
    del temp_data_for_dim

    encoder_parameters = {
        'input_dim': encoder_input_dim_placeholder,
        'hidden_dim': 128,
        'output_dim': gcn_output_dim,
        'num_layers': 3,
        'dropout': 0.5
    }

    decoder_parameters = {
        'hidden_dim': 128,
        'num_layers': 8,
        'dropout': 0.3
    }

    training_parameters = {
        'val_ratio': 0.1,
        'test_ratio': 0.1,
        'lr': 0.001,
        'epochs': 1000,
        'log_every_epochs': 10
    }

    trained_encoder_model, trained_decoder_model, final_test_f1_score = train_link_prediction_model(
        pyg_data_path=pyg_data_file,
        encoder_params=encoder_parameters,
        decoder_params=decoder_parameters,
        training_params=training_parameters
    )

    print(f"\n训练流程已完成。最终测试 F1: {final_test_f1_score:.4f}")


