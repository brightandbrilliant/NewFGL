import numpy as np

# pFGL 数据
pFGL_recall = [0.8602, 0.8556, 0.8811, 0.8750, 0.9176, 0.8579, 0.8999, 0.8934, 0.8768, 0.8107]
pFGL_precision = [0.9197, 0.9099, 0.8827, 0.9019, 0.8559, 0.9023, 0.8689, 0.8874, 0.8945, 0.8883]
pFGL_F1 = [0.8889, 0.8819, 0.8818, 0.8881, 0.8854, 0.8796, 0.8841, 0.8904, 0.8856, 0.8440]

# FedAVG 数据
FedAVG_recall = [0.8857, 0.8837, 0.8871, 0.8718, 0.8721, 0.8860, 0.8774, 0.8790, 0.8910, 0.8842]
FedAVG_precision = [0.8952, 0.8819, 0.8854, 0.9056, 0.9011, 0.8881, 0.8899, 0.8919, 0.8818, 0.8862]
FedAVG_F1 = [0.8904, 0.8828, 0.8862, 0.8884, 0.8863, 0.8871, 0.8836, 0.8854, 0.8864, 0.8852]

# Cluster_Version_Simple 数据
Cluster_recall = [0.9253, 0.9220, 0.8935, 0.9025, 0.9028, 0.9098, 0.9240, 0.9011, 0.8914, 0.9340]
Cluster_precision = [0.8467, 0.8384, 0.8780, 0.8730, 0.8657, 0.8657, 0.8348, 0.8710, 0.8790, 0.8197]
Cluster_F1 = [0.8842, 0.8782, 0.8855, 0.8874, 0.8839, 0.8872, 0.8772, 0.8858, 0.8851, 0.8730]

def calc_avg(data):
    return round(np.mean(data), 4)

# 计算 pFGL
Cluster = {
    "Recall": calc_avg(Cluster_recall),
    "Precision": calc_avg(Cluster_precision),
    "F1": calc_avg(Cluster_F1)
}
Cluster["Overall"] = calc_avg(list(Cluster.values()))

# 计算 FedAVG (同上逻辑)
# 计算 Cluster_Version_Simple (同上逻辑)

print(Cluster)
