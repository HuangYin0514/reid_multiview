import torch
import torch.nn as nn
import torch.nn.functional as F

from ..module import weights_init


class SharedSpecialLoss(nn.Module):
    def __init__(self):
        super(SharedSpecialLoss, self).__init__()

    def cosine_similarity(self, embedded_a, embedded_b):
        embedded_a = F.normalize(embedded_a, dim=1)
        embedded_b = F.normalize(embedded_b, dim=1)
        sim = torch.matmul(embedded_a, embedded_b.T)
        return torch.clamp(sim, min=0.0005, max=0.9995)

    def distance(self, embedded_a, embedded_b):
        sim = self.cosine_similarity(embedded_a, embedded_b)
        loss = -torch.log(1 - sim)
        return torch.mean(loss)

    def forward(self, shared_features, specific_features):
        num_views = 4
        batch_size = shared_features.size(0)
        chunk_size = batch_size // num_views

        shared_specific_loss = 0
        for i in range(chunk_size):
            shared_features_chunk = shared_features[num_views * i : num_views * (i + 1), ...]
            specific_features_chunk = specific_features[num_views * i : num_views * (i + 1), ...]

            # Loss between shared and specific features
            shared_specific_loss += self.distance(shared_features_chunk, specific_features_chunk)

        return shared_specific_loss


class SharedSharedLoss(nn.Module):
    def __init__(self):
        super(SharedSharedLoss, self).__init__()

        margin = 0.3
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def self_euclidean_similarity(self, inputs):
        # Compute pairwise distance, replace by the official when merged
        # ||a-b||^2 = ||a||^2 -2 * <a,b> + ||b||^2
        n = inputs.size(0)
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist = torch.addmm(input=dist, mat1=inputs, mat2=inputs.t(), alpha=-2, beta=1)
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        return dist

    def distance(self, embedded_a):
        bs = embedded_a.shape[0]
        embedded_a = F.normalize(embedded_a, dim=1)
        sims = self.self_euclidean_similarity(embedded_a)

        mask = ~torch.eye(bs, dtype=torch.bool)
        non_diag_sims = sims[mask]  # 取出非对角线的值
        max_dist, _ = torch.max(non_diag_sims.view(bs, -1), dim=1)  # 找到距离最远的组

        y = torch.ones_like(max_dist)
        dist_an = torch.zeros_like(max_dist)
        dist_ap = max_dist
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss

    def forward(self, shared_features):
        num_views = 4
        batch_size = shared_features.size(0)
        chunk_size = batch_size // num_views

        shared_consistency_loss = 0
        for i in range(chunk_size):
            shared_features_chunk = shared_features[num_views * i : num_views * (i + 1), ...]

            # Loss within shared features
            shared_consistency_loss += self.distance(shared_features_chunk)

        return shared_consistency_loss


class DecouplingLoss(nn.Module):
    def __init__(self, config):
        super(DecouplingLoss, self).__init__()
        self.config = config

    def forward(self, shared_features, specific_features):
        SharedSpecial_loss = SharedSpecialLoss().forward(shared_features, specific_features)
        SharedShared_loss = SharedSharedLoss().forward(shared_features)
        loss = SharedSpecial_loss + 0.01 * SharedShared_loss
        return loss


#################################################################
# network
#################################################################


class FeatureDecouplingNet(nn.Module):
    """
    特征解耦模块，用于将共享特征和特定特征进行解耦。

    该模块接收共享特征和特定特征，并通过某种方式对它们进行处理，以实现特征解耦。

    参数:
    nn.Module (torch.nn.Module): 继承自 PyTorch 的 nn.Module 类。

    方法:
    forward(shared_features, specific_features):
        前向传播方法，接收共享特征和特定特征，并返回解耦后的特征。
    """

    def __init__(self, config):
        super(FeatureDecouplingNet, self).__init__()
        self.config = config

        # shared branch
        ic = 2048
        oc = 1024
        self.mlp1 = nn.Sequential(
            nn.Linear(ic, oc, bias=False),
            nn.BatchNorm1d(oc),
        )
        self.mlp1.apply(weights_init.weights_init_kaiming)

        # special branch
        self.mlp2 = nn.Sequential(
            nn.Linear(ic, oc, bias=False),
            nn.BatchNorm1d(oc),
        )
        self.mlp2.apply(weights_init.weights_init_kaiming)

    def forward(self, features):
        shared_features = self.mlp1(features)
        special_features = self.mlp2(features)
        return shared_features, special_features


class FeatureIntegrationNet(nn.Module):
    def __init__(self, config):
        super(FeatureIntegrationNet, self).__init__()
        self.config = config

        # shared branch
        ic = 1024 * 4
        oc = 1024
        self.mlp1 = nn.Sequential(
            nn.Linear(ic, oc, bias=False),
            nn.BatchNorm1d(oc),
            nn.Linear(oc, oc, bias=False),
            nn.BatchNorm1d(oc),
        )
        self.mlp1.apply(weights_init.weights_init_kaiming)

    def __call__(self, features, pids):
        size = features.size(0)
        c = features.size(1)
        chunk_size = int(size / 4)  # 16

        integrate_features = torch.zeros([chunk_size, c * 4]).to(features.device)
        integrate_pids = torch.zeros([chunk_size]).to(pids.device)

        for i in range(chunk_size):
            integrate_features[i] = torch.cat([features[4 * i].unsqueeze(0), features[4 * i + 1].unsqueeze(0), features[4 * i + 2].unsqueeze(0), features[4 * i + 3].unsqueeze(0)], dim=1)
            integrate_pids[i] = pids[4 * i]

        integrate_features = self.mlp1(integrate_features)

        return integrate_features, integrate_pids
