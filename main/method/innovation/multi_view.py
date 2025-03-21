import torch
from torch.nn import functional as F


class FeatureIntegration:
    """
    FeatureIntegration 类用于将连续的特征进行集成处理。

    主要功能：
      - 将输入特征按照一定的分组策略（每4个样本一组）进行求和融合，
        得到集成后的特征表示。
      - 同时根据 pids 进行相应的采样，确保融合后的特征与身份标签相匹配。

    参数:
        config: 模型配置参数，用于设置特征集成中的相关参数。

    调用:
        输入:
            - features: tensor，输入特征，形状为 [batch_size, feature_dim]
            - pids: tensor，对应的身份 ID，长度为 batch_size
        输出:
            - integrating_features: tensor，集成后的特征，形状为 [batch_size/4, feature_dim]
            - integrating_pids: tensor，采样后的 pids，形状为 [batch_size/4]
    """

    def __init__(self, config):
        super(FeatureIntegration, self).__init__()
        self.config = config

    def __call__(self, features, pids):
        size = features.size(0)
        chunk_size = int(size / 4)  # 16
        c = features.size(1)

        integrating_features = torch.zeros([chunk_size, c]).to(features.device)
        integrating_pids = torch.zeros([chunk_size], dtype=torch.long).to(pids.device)

        for i in range(chunk_size):
            integrating_features[i] = 1 * (features[4 * i] + features[4 * i + 1] + features[4 * i + 2] + features[4 * i + 3])
            integrating_pids[i] = pids[4 * i]

        return integrating_features, integrating_pids


class ContrastLoss:
    """
    ContrastLoss 类用于计算特征之间的对比损失。

    本类提供三种计算损失的方法：
      1. __call__: 默认调用方法，通过将 features_2 重复扩展4次，
         然后计算 features_1 与扩展后的 features_2 之间的 L2 范数损失。
      2. v1: 手动构造扩展后的 features_2，然后计算两组特征之间的 L2 范数损失。包含正则损失。
      3. v2: 计算 features_1 自身的 L2 范数损失，用于辅助或简化计算。

    参数:
        config: 模型配置参数，用于设置损失计算中的相关参数。
    """

    def __init__(self, config):
        super(ContrastLoss, self).__init__()
        self.config = config

    def __call__(self, features_1, features_2):
        new_features_2 = torch.zeros(features_1.size()).to(features_1.device)
        for i in range(int(features_2.size(0) / 4)):
            new_features_2[i * 4 : i * 4 + 4] = features_2[i]
        loss = torch.norm((features_1 - new_features_2), p=2)
        return loss

    # def __call__(self, features_1, features_2):
    #     new_features_2 = torch.repeat_interleave(features_2, repeats=4, dim=0).clone().detach()
    #     loss = torch.norm((features_1 - new_features_2), p=2)
    #     return loss

    # def v1(self, features_1, features_2):
    #     # v463
    #     new_features_2 = torch.zeros(features_1.size()).to(features_1.device)
    #     for i in range(int(new_features_2.size(0) / 4)):
    #         new_features_2[i * 4 : i * 4 + 4] = features_2[i]
    #     loss = torch.norm((features_1 - new_features_2), p=2)
    #     return loss

    # def v2(self, features_1, features_2):
    #     # v462
    #     loss = torch.norm(features_1, p=2)
    #     return loss


class FeatureMapLocation:
    """
    FeatureMapLocation

    该类用于根据输入的特征图、pids和分类器生成局部化后的特征图。
    利用分类器最后一层的参数对特征图进行加权，生成归一化后的热图，
    并通过热图对原始特征图进行局部调整，从而增强关键区域的特征表达。

    参数:
      config: 用于初始化类的配置信息

    调用:
      输入:
        - features_map: 原始的特征图
        - pids: 对应的身份标识
        - classifier: 模型分类器，用于获取参数计算热图

      输出:
        - 返回局部化处理后的特征图
    """

    def __init__(self, config):
        super(FeatureMapLocation, self).__init__()
        self.config = config

    def __call__(self, features_map, pids, classifier):
        size = features_map.size(0)
        chunk_size = int(size / 4)
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
