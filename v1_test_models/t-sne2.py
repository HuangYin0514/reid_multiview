import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

# 使用numpy生成模拟数据
np.random.seed(42)
n_samples = 300
n_features = 2048
n_classes = 10

# 创建三个随机分布的簇
X1 = np.random.randn(n_samples, n_features)
X2 = np.random.randn(n_samples, n_features)
X1_label = np.array([i for i in range(n_classes) for _ in range(n_samples // n_classes)])
X2_label = np.array([i for i in range(n_classes) for _ in range(n_samples // n_classes)])

x_cat = np.concatenate((X1, X2), axis=0)
y_cat = np.concatenate((X1_label, X2_label), axis=0)

# 可视化结果
plot_list = [3, 7, 8]
plt.figure(figsize=(8, 5))
for i in np.unique(y_cat):
    if i not in plot_list:
        continue
    plt.scatter(x_cat[y_cat == i, 0], x_cat[y_cat == i, 1], label=f"Class {i}")

mask = np.isin(X1_label, plot_list)
plt.scatter(x_cat[:n_samples][mask, 0], x_cat[:n_samples][mask, 1], s=100, c="red", marker="x", alpha=0.8, label="X1 Marked")

plt.title("t-SNE Visualization of Simulated Data with Numpy")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.legend()
plt.savefig("tsne_visualization.png")  # 保存图像
plt.show()
