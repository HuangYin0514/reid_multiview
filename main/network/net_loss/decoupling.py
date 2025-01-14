import torch
import torch.nn as nn
import torch.nn.functional as F

from .similarity import compute_cosine_similarity, compute_self_euclidean_distance


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
