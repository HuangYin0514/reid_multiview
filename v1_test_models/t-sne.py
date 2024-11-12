import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

# 使用numpy生成模拟数据
np.random.seed(42)
n_samples = 300
n_features = 2048
n_classes = 3

# 创建三个随机分布的簇
X = np.vstack([np.random.randn(n_samples // n_classes, n_features) + np.array([i * 5] + [0] * (n_features - 1)) for i in range(n_classes)])
y = np.array([i for i in range(n_classes) for _ in range(n_samples // n_classes)])

print(np.array([1 * 5] + [0] * (n_features - 1)))
print(X.shape, y.shape)

# 使用t-SNE进行降维
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# 可视化结果
plt.figure(figsize=(8, 5))
for i in np.unique(y):
    plt.scatter(X_tsne[y == i, 0], X_tsne[y == i, 1], label=f"Class {i}")
plt.title("t-SNE Visualization of Simulated Data with Numpy")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.legend()
plt.savefig("tsne_visualization.png")  # 保存图像
plt.show()
