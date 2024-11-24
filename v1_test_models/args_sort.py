import numpy as np

# 示例分数矩阵
scores = np.array([[0.2, 0.8, 0.5], [0.6, 0.4, 0.9], [0.1, 0.3, 0.2]])

# 对每一行排序，并返回索引
rank_results = np.argsort(scores)[:, ::-1]

print("Scores:")
print(scores)
print("\nRanked indices (descending order):")
print(rank_results)
