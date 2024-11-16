import torch

# 创建一个 4x4 的张量
A = torch.rand(4, 4)

# 创建一个掩码：对角线上的元素为 False，非对角线上的元素为 True
mask = ~torch.eye(4, dtype=torch.bool)

# 应用掩码来获取非对角线元素
non_diag_elements = A[mask]

print(A)
print(non_diag_elements)
