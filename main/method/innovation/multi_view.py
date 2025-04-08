import torch
import torch.nn as nn
from torch.nn import functional as F


class MultiviewFeatureFusion(nn.Module):

    def __init__(self, view_num):
        super(MultiviewFeatureFusion, self).__init__()
        self.view_num = view_num

    def forward(self, features, pids):
        size = features.size(0)
        chunk_size = int(size / self.view_num)  # 16
        C = features.size(1)

        # Reshape: [batch_size, C] -> [chunk_size, view_num, C]
        fused = features.view(chunk_size, self.view_num, C)
        fusion_features = fused.sum(dim=1)  # or .mean(dim=1)

        # 保留每组的第一个pid [size] -> [chunk_size]
        fusion_pids = pids.view(chunk_size, self.view_num)[:, 0]

        return fusion_features, fusion_pids


class FeatureQuantification(nn.Module):

    def __init__(self, view_num):
        super(FeatureQuantification, self).__init__()
        self.view_num = view_num

    def forward(self, features, cls_scores, pids):
        size = features.size(0)
        prob = torch.log_softmax(cls_scores, dim=1)
        probs = prob[torch.arange(size), pids]
        weights = torch.softmax(probs.view(-1, self.view_num), dim=1).view(-1).clone().detach()
        quantified_features = weights.unsqueeze(1) * features  #  注意：调整weight的维度
        return quantified_features


class FeatureMapLocation(nn.Module):

    def __init__(self):
        super(FeatureMapLocation, self).__init__()

    def forward(self, features_map, pids, classifier):
        size = features_map.size(0)
        c, h, w = features_map.size(1), features_map.size(2), features_map.size(3)
        localized_features_map = torch.zeros([size, c, h, w]).to(features_map.device)

        heatmaps = torch.zeros((size, h, w), device=features_map.device)
        for i in range(size):
            classifier_name = []
            classifier_params = []
            for name, param in classifier.named_parameters():
                classifier_name.append(name)
                classifier_params.append(param)
            heatmap_i = torch.matmul(classifier_params[-1][pids[i]].unsqueeze(0), features_map[i].unsqueeze(0).reshape(c, h * w)).detach()
            if heatmap_i.max() != 0:
                heatmap_i = (heatmap_i - heatmap_i.min()) / (heatmap_i.max() - heatmap_i.min())
            heatmap_i = heatmap_i.reshape(h, w)
            heatmap_i = torch.tensor(heatmap_i)
            heatmaps[i, :, :] = heatmap_i

        localized_features_map = features_map * heatmaps.unsqueeze(1).clone().detach()

        return localized_features_map


class ContrastLoss(nn.Module):
    def __init__(self, view_num):
        super(ContrastLoss, self).__init__()
        self.view_num = view_num

    def forward(self, features_1, features_2):
        bs = features_1.size(0)
        chunk_bs = int(bs / self.view_num)

        new_features_2 = torch.zeros(features_1.size()).to(features_1.device)
        for i in range(int(chunk_bs / self.view_num)):
            new_features_2[i * self.view_num : i * self.view_num + self.view_num] = features_2[i]
        # loss = 0.448 / bs * torch.norm((features_1 - new_features_2), p=2)
        loss = 0.007 * torch.norm((features_1 - new_features_2.detach()), p=2)
        return loss
