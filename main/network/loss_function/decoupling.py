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


class SharedSpecialLoss(nn.Module):
    def __init__(self):
        super(SharedSpecialLoss, self).__init__()

    def forward(self, embedded_a, embedded_b):
        sim = compute_cosine_similarity(embedded_a, embedded_b)
        loss = -torch.log(1 - sim)
        return torch.mean(loss)


class SharedSharedLoss(nn.Module):
    def __init__(self):
        super(SharedSharedLoss, self).__init__()

        margin = 0.3
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, embedded_a):
        embedded_a = F.normalize(embedded_a, dim=1)
        sims = compute_self_euclidean_distance(embedded_a)
        bs = embedded_a.shape[0]
        mask = ~torch.eye(bs, dtype=torch.bool)  # mask out diagonal
        non_diag_sims = sims[mask]

        # 找到距离最远的组
        max_dist, _ = torch.max(non_diag_sims.view(bs, -1), dim=1)

        # 计算损失
        dist_an = torch.zeros_like(max_dist)
        dist_ap = max_dist
        y = torch.ones_like(max_dist)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss


class SpecificSpecificLoss(nn.Module):
    def __init__(self):
        super(SpecificSpecificLoss, self).__init__()

        margin = 0.3
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, embedded_a):
        sim = compute_cosine_similarity(embedded_a, embedded_a)
        loss = -torch.log(1 - sim)
        return torch.mean(loss)


class DecouplingSharedSpecialLoss(nn.Module):
    def __init__(self):
        super(DecouplingSharedSpecialLoss, self).__init__()

    def forward(self, shared_features, specific_features):
        num_views = 4
        batch_size = shared_features.size(0)
        chunk_size = batch_size // num_views

        shared_specific_loss = 0
        for i in range(chunk_size):
            shared_features_chunk = shared_features[num_views * i : num_views * (i + 1), ...]
            specific_features_chunk = specific_features[num_views * i : num_views * (i + 1), ...]

            # Loss between shared and specific features
            shared_specific_loss += SharedSpecialLoss().forward(shared_features_chunk, specific_features_chunk)

        return shared_specific_loss


class DecouplingSharedSharedLoss(nn.Module):
    def __init__(self):
        super(DecouplingSharedSharedLoss, self).__init__()

    def forward(self, shared_features):
        num_views = 4
        batch_size = shared_features.size(0)
        chunk_size = batch_size // num_views

        shared_consistency_loss = 0
        for i in range(chunk_size):
            shared_features_chunk = shared_features[num_views * i : num_views * (i + 1), ...]

            # Loss within shared features
            shared_consistency_loss += SharedSharedLoss().forward(shared_features_chunk)

        return shared_consistency_loss


class DecouplingSpecificSpecificLoss(nn.Module):
    def __init__(self):
        super(DecouplingSpecificSpecificLoss, self).__init__()

    def forward(self, specific_features):
        num_views = 4
        batch_size = specific_features.size(0)
        chunk_size = batch_size // num_views

        specific_consistency_loss = 0
        for i in range(chunk_size):
            specific_features_chunk = specific_features[num_views * i : num_views * (i + 1), ...]

            # Loss within specific features
            specific_consistency_loss += SpecificSpecificLoss().forward(specific_features_chunk)

        return specific_consistency_loss
