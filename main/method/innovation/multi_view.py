import torch
import torch.nn as nn
from torch.nn import functional as F

from .. import module


class MultiviewFeatureFusion(nn.Module):
    """
    多视图特征融合模块，用于将多个视图的特征进行融合。

    该模块包含两个核心功能：
    1. 特征融合：将两个输入特征进行平均融合。
    2. 视图融合：将融合后的特征按照视图进行平均或求和。

    参数:
    - view_num (int): 视图数量。
    - input_dim (int): 输入特征的维度。
    - out_dim (int): 输出特征的维度。

    示例:
    >>> model = MultiviewFeatureFusion(view_num=2, input_dim=128, out_dim=64)
    >>> features_1 = torch.randn(32, 128)
    >>> features_2 = torch.randn(32, 128)
    >>> pids = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
    >>> fused_features, fused_pids = model(features_1, features_2, pids)
    """

    def __init__(self, view_num, input_dim, out_dim):
        super(MultiviewFeatureFusion, self).__init__()
        self.view_num = view_num

        # self.fusion_module = module.Residual(
        #     nn.Sequential(
        #         nn.Conv1d(input_dim, out_dim, kernel_size=1),
        #         nn.ReLU(),
        #         nn.BatchNorm1d(out_dim),
        #         nn.Conv1d(out_dim, out_dim, kernel_size=1),
        #     )
        # )

        self.fusion_module = module.Residual(
            nn.Sequential(
                nn.Conv1d(input_dim, out_dim, kernel_size=1),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(),
            )
        )

        self._initialize_weights()

    def forward(self, features_1, features_2, pids):
        B, C = features_1.shape  # batch size, channels
        chunk_size = B // self.view_num

        # Feature-level fusion
        fused = ((features_1 + features_2) / 2).unsqueeze(-1)  # [B, C, 1]
        fused = self.fusion_module(fused).squeeze(-1)  # [B, C]

        # View-level fusion: reshape to [chunk_size, view_num, C]
        fused = fused.view(chunk_size, self.view_num, C)
        fused = fused.mean(dim=1)  # [chunk_size, C]

        # PID-level fusion: assume same identity across views
        fused_pids = pids.view(chunk_size, self.view_num)[:, 0]

        return fused, fused_pids

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_in")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                if m.affine:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)


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

        self.distillKL = DistillKL(T)

    def forward(self, features_1_logits, features_2_logits, features_1, features_2):

        loss = 0.0
        # -------------- regularization  --------------
        # normed_features_1 = F.normalize(features_1, p=2, dim=1)
        # reg_loss = 0.007 * torch.norm(features_1, p=2)
        # loss += reg_loss
        # -------------- Contrastive loss  --------------
        # new_features_2 = torch.repeat_interleave(features_2, self.view_num, dim=0)  # [batch_size, c] -> [batch_size * view_num, c]
        # input1_normed = F.normalize(features_1, p=2, dim=1)
        # input2_normed = F.normalize(new_features_2, p=2, dim=1)
        # reg_loss = 0.05 * torch.norm((input1_normed - input2_normed), p=2)
        # loss += reg_loss
        # -------------- KL loss  --------------
        new_features_2_logits = torch.repeat_interleave(features_2_logits, self.view_num, dim=0)  # [batch_size, c] -> [batch_size * view_num]
        kl_loss = 0.5 * (self.distillKL(features_1_logits, new_features_2_logits) + self.distillKL(new_features_2_logits, features_1_logits))
        loss += kl_loss
        return loss
