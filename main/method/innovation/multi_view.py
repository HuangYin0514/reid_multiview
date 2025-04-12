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

        # Reshape: [batch_size, C] -> [chunk_size, view_num, C] -> [chunk_size, C]
        fused = features.view(chunk_size, self.view_num, C)
        fusion_features = fused.mean(dim=1)  # or .sum(dim=1)

        # 保留每组的第一个pid [size] -> [chunk_size]
        fusion_pids = pids.view(chunk_size, self.view_num)[:, 0]

        return fusion_features, fusion_pids

    """
    def forward(self, features, pids):
        size = features.size(0)
        chunk_size = int(size / self.view_num)  # 16
        c = features.size(1)

        integrating_features = torch.zeros([chunk_size, c]).to(features.device)
        integrating_pids = torch.zeros([chunk_size]).to(pids.device)

        for i in range(chunk_size):
            integrating_features[i] = 1 * (
                features[self.view_num * i] + features[self.view_num * i + 1] + features[self.view_num * i + 2] + features[self.view_num * i + 3]
            )
            integrating_pids[i] = pids[self.view_num * i]

        return integrating_features, integrating_pids
    """


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
        size, c, h, w = features_map.size()

        # 提前提取分类器最后一层的权重
        with torch.no_grad():
            classifier_weight = list(classifier.parameters())[-1]  # shape: [num_classes, c]
            selected_weights = classifier_weight[pids]  # shape: [batch_size, c]

            # 计算热力图
            features_map_flat = features_map.view(size, c, h * w)  # [batch, c, h*w]
            heatmaps = torch.bmm(selected_weights.unsqueeze(1), features_map_flat)  # [batch, 1, h*w]
            heatmaps = heatmaps.view(size, h, w)

            # 归一化
            heatmaps_min = heatmaps.view(size, -1).min(dim=1, keepdim=True)[0].view(size, 1, 1)
            heatmaps_max = heatmaps.view(size, -1).max(dim=1, keepdim=True)[0].view(size, 1, 1)
            heatmaps = (heatmaps - heatmaps_min) / (heatmaps_max - heatmaps_min + 1e-6)  # 避免除以0

        # 应用热力图
        localized_features_map = features_map * heatmaps.unsqueeze(1)

        return localized_features_map


class ContrastLoss(nn.Module):
    def __init__(self, view_num):
        super(ContrastLoss, self).__init__()
        self.view_num = view_num

    def forward(self, features_1, features_2):
        # new_features_22 = torch.zeros(features_1.size()).to(features_1.device)
        # for i in range(int(features_1.size(0) / 4)):
        #     new_features_22[i * 4 : i * 4 + 4] = features_2[i]
        new_features_2 = torch.repeat_interleave(features_2, self.view_num, dim=0)  # [batch_size, c] -> [batch_size * view_num, c]
        input1_normed = F.normalize(features_1, p=2, dim=1)
        input2_normed = F.normalize(new_features_2, p=2, dim=1)
        loss = 0.05 * torch.norm((input1_normed - input2_normed), p=2)
        return loss


class DistillKL(nn.Module):
    """KL divergence for distillation"""

    def __init__(self, T=4):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, features_1_logits, features_2_logits):
        p_s = F.log_softmax(features_1_logits / self.T, dim=1)
        p_t = F.softmax(features_2_logits / self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / features_1_logits.shape[0]
        return loss


class MVDistillKL(nn.Module):
    """KL divergence for distillation"""

    def __init__(self, view_num, T=4):
        super(MVDistillKL, self).__init__()
        self.T = T
        self.view_num = view_num

    def forward(self, features_1_logits, features_2_logits):
        new_features_2_logits = torch.repeat_interleave(features_2_logits, self.view_num, dim=0)  # [batch_size, c] -> [batch_size * view_num]
        p_s = F.log_softmax(features_1_logits / self.T, dim=1)
        p_t = F.softmax(new_features_2_logits / self.T, dim=1)
        loss = 0.1 * F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / features_1_logits.shape[0]
        return loss
