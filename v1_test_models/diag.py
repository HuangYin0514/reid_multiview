import torch

# 创建一个 5x5 的矩阵
A = torch.randn(5, 5)

# 创建一个对角线掩码
mask = torch.eye(5, dtype=torch.bool)

# 使用掩码来获取非对角线元素
non_diag_elements = A[~mask]  # 取出非对角线的元素

print("矩阵 A:")
print(A)
print("\n非对角线元素:")
print(non_diag_elements)
