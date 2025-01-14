import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_cosine_similarity(embedded_a, embedded_b):
    embedded_a = F.normalize(embedded_a, dim=1)
    embedded_b = F.normalize(embedded_b, dim=1)
    sim = torch.matmul(embedded_a, embedded_b.T)
    return torch.clamp(sim, min=0.0005, max=0.9995)


def compute_self_euclidean_distance(inputs):
    """
    计算输入张量中每对样本之间的欧氏距离。

    参数:
    inputs (torch.Tensor): 输入张量，形状为 (n, d)，其中 n 是样本数量, d 是特征维度。

    返回:
    torch.Tensor: 输出张量，形状为 (n, n)，其中每个元素表示对应样本对之间的欧氏距离。
    """
    n = inputs.size(0)

    # Compute pairwise distance, replace by the official when merged
    # ||a-b||^2 = ||a||^2 -2 * <a,b> + ||b||^2
    dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
    dist = dist + dist.t()
    # dist.addmm_(1, -2, inputs, inputs.t())
    dist = torch.addmm(input=dist, mat1=inputs, mat2=inputs.t(), alpha=-2, beta=1)
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist
