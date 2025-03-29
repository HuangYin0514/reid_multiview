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


#################################################################
# network
#################################################################


class FeatureDecouplingModule(nn.Module):
    def __init__(self, config):
        super(FeatureDecouplingModule, self).__init__()
        self.config = config

        #################################################################
        # shared branch
        ic = 2048
        oc = 1024
        self.decoupling_mlp1 = nn.Sequential(
            nn.Linear(ic, oc, bias=False),
            nn.BatchNorm1d(oc),
        )
        self.decoupling_mlp1.apply(weights_init.weights_init_kaiming)

        # special branch
        self.decoupling_mlp2 = nn.Sequential(
            nn.Linear(ic, oc, bias=False),
            nn.BatchNorm1d(oc),
        )
        self.decoupling_mlp2.apply(weights_init.weights_init_kaiming)

        #################################################################
        # Reconstruction branch
        ic = 1024 * 2
        oc = 2048
        self.reconstructed_mlp = nn.Sequential(
            nn.Linear(ic, oc, bias=False),
            nn.BatchNorm1d(oc),
            nn.ReLU(inplace=True),
            nn.Linear(oc, oc, bias=False),
        )
        self.reconstructed_mlp.apply(weights_init.weights_init_kaiming)

    def encoder(self, features):
        shared_features = self.decoupling_mlp1(features)
        special_features = self.decoupling_mlp2(features)
        return shared_features, special_features

    def decoder(self, shared_features, special_features):
        reconstructed_features = torch.cat([shared_features, special_features], dim=1)
        reconstructed_features = self.reconstructed_mlp(reconstructed_features)
        return reconstructed_features

    def forward(self, features):
        shared_features, special_features = self.encoder(features)
        reconstructed_features = self.decoder(shared_features, special_features)
        return shared_features, special_features, reconstructed_features


######################################################################
class FeatureFusionModule(nn.Module):
    def __init__(self, config):
        super(FeatureFusionModule, self).__init__()
        self.config = config
        self.num_views = 4

        ic = 1024 * 2
        oc = 2048
        self.fusion_mlp = nn.Sequential(
            nn.Linear(ic, oc, bias=False),
            nn.BatchNorm1d(oc),
        )
        self.fusion_mlp.apply(weights_init.weights_init_kaiming)

        self.ic = ic
        self.oc = oc

    def forward(self, shared_features, special_features, pids):
        bs = shared_features.size(0)
        chunk_bs = int(bs / self.num_views)  # 16
        dim = shared_features.size(1)

        fusion_features_temp = torch.cat([shared_features, special_features], dim=1)

        fusion_features = torch.zeros([chunk_bs, dim]).to(shared_features.device)
        fusion_pids = torch.zeros([chunk_bs], dtype=torch.int).to(pids.device)
        for i in range(chunk_bs):
            fusion_features[i] = fusion_features_temp[4 * i] + fusion_features_temp[4 * i + 1] + fusion_features_temp[4 * i + 2] + fusion_features_temp[4 * i + 3]
            fusion_pids[i] = pids[4 * i]
        # fusion_features = self.fusion_mlp(fusion_features)
        return fusion_features, fusion_pids


######################################################################
class FeatureViewsFusionModule(nn.Module):
    def __init__(self, config):
        super(FeatureViewsFusionModule, self).__init__()
        self.config = config
        self.num_views = 4

    def forward(self, features, pids):
        bs = features.size(0)
        chunk_bs = int(bs / self.num_views)  # 16
        dim = features.size(1)

        fusion_features = torch.zeros([chunk_bs, dim]).to(features.device)
        fusion_pids = torch.zeros([chunk_bs], dtype=torch.int).to(pids.device)
        for i in range(chunk_bs):
            fusion_features[i] = features[4 * i] + features[4 * i + 1] + features[4 * i + 2] + features[4 * i + 3]
            fusion_pids[i] = pids[4 * i]

        # fusion_features = self.fusion_mlp(fusion_features)
        return fusion_features, fusion_pids


######################################################################
# class FeatureFusionModule(nn.Module):
#     def __init__(self, config):
#         super(FeatureFusionModule, self).__init__()
#         self.config = config
#         self.num_views = 4

#         ic = 512 * 2
#         oc = 2048
#         self.fusion_mlp = nn.Sequential(
#             nn.Linear(ic, oc, bias=False),
#             nn.BatchNorm1d(oc),
#         )
#         self.fusion_mlp.apply(weights_init.weights_init_kaiming)

#         self.ic = ic
#         self.oc = oc

#     def forward(self, shared_features, special_features, pids):
#         bs = shared_features.size(0)
#         chunk_bs = int(bs / self.num_views)  # 16

#         fusion_features_list = []
#         fusion_pids_list = []
#         for i in range(chunk_bs):
#             shared_features_item = (shared_features[2 * i] + shared_features[2 * i + 1]).unsqueeze(0)
#             special_features_item = (special_features[2 * i] + special_features[2 * i + 1]).unsqueeze(0)
#             fusion_features_list.append(torch.cat([shared_features_item, special_features_item], dim=1))
#             fusion_pids_list.append(pids[2 * i].unsqueeze(0))
#         fusion_features = torch.cat(fusion_features_list, dim=0)  # (chunk_bs, shared_dim + 4*special_dim)
#         fusion_pids = torch.cat(fusion_pids_list, dim=0)  # (chunk_bs,)

#         fusion_features = torch.cat([shared_features, special_features], dim=1)
#         fusion_features = self.fusion_mlp(fusion_features)
#         return fusion_features, fusion_pids

######################################################################
# class FeatureFusionModule(nn.Module):
#     def __init__(self, config):
#         super(FeatureFusionModule, self).__init__()
#         self.config = config
#         self.num_views = 4

#         ic = 512 * 2
#         oc = 2048
#         self.fusion_mlp = nn.Sequential(
#             nn.Linear(ic, oc, bias=False),
#             nn.BatchNorm1d(oc),
#         )
#         self.fusion_mlp.apply(weights_init.weights_init_kaiming)

#         self.ic = ic
#         self.oc = oc

#     def forward(self, shared_features, special_features, pids):
#         bs = shared_features.size(0)
#         chunk_bs = int(bs / self.num_views)  # 16
#         fusion_features = torch.cat([shared_features, special_features], dim=1)
#         fusion_features = self.fusion_mlp(fusion_features)
#         return fusion_features, pids
