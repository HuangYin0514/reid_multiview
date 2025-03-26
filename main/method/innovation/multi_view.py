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
        dim = features.size(1)

        integrating_features = torch.zeros([chunk_size, dim]).to(features.device)
        integrating_pids = torch.zeros([chunk_size]).to(pids.device)

        for i in range(chunk_size):
            integrating_features[i] = 1 * (features[4 * i] + features[4 * i + 1] + features[4 * i + 2] + features[4 * i + 3])
            integrating_pids[i] = pids[4 * i]

        return integrating_features, integrating_pids


class ContrastLoss:
    """
    ContrastLoss 类用于计算特征之间的对比损失。

    参数:
        config: 模型配置参数，用于设置损失计算中的相关参数。
    """

    def __init__(self, config):
        super(ContrastLoss, self).__init__()
        self.config = config

    def __call__(self, features_1, features_2):
        bs = features_1.size(0)
        # # method 1
        # loss = torch.norm(features_1, p=2)

        # # method 2
        # new_features_2 = torch.repeat_interleave(features_2, repeats=4, dim=0).clone().detach()
        # loss = 0.5 * torch.nn.MSELoss(reduction="mean")(features_1, new_features_2)

        # method 3
        # new_features_2 = torch.zeros(features_1.size()).to(features_1.device)
        # for i in range(int(features_2.size(0))):
        #     new_features_2[i * 4 : i * 4 + 4] = features_2[i]
        # loss = 1 * torch.nn.MSELoss(reduction="mean")(features_1, new_features_2)

        # method 4
        # new_features_2 = torch.zeros(features_1.size()).to(features_1.device)
        # for i in range(int(features_2.size(0))):
        #     new_features_2[i * 4 : i * 4 + 4] = features_2[i]
        # loss = torch.nn.MSELoss(reduction="mean")(features_1, new_features_2)  # 1/B * ||f1 - f2||^2_F

        # method 5
        new_features_2 = torch.zeros(features_1.size()).to(features_1.device)
        for i in range(int(features_2.size(0) / 4)):
            new_features_2[i * 4 : i * 4 + 4] = features_2[i]
        loss = torch.nn.MSELoss(reduction="mean")(features_1, new_features_2)  # 1/B * ||f1 - f2||^2_F

        return loss


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


class FeatureQuantification:
    """
    FeatureQuantification 类用于对输入的特征进行量化处理。

    功能描述：
    - 根据模型输出的分类分数(cls_scores)计算每个样本对应类别的对数概率。
    - 利用pids从对数概率中选择对应的类别概率，并进一步通过 softmax 计算权重。
    - 利用权重对输入特征(features)进行加权，得到量化后的特征表示。

    参数:
        config: 类的配置信息，用于初始化相关设置。

    调用:
        输入:
            - features: 输入特征张量，形状为 [batch_size, feature_dim]
            - cls_scores: 模型输出的分类分数，形状为 [batch_size, num_classes]
            - pids: 对应的身份标识，长度为 batch_size，表示每个样本对应的类别索引
        输出:
            - quantified_features: 量化后的特征张量，形状与 features 相同
    """

    def __init__(self, config):
        super(FeatureQuantification, self).__init__()
        self.config = config
        self.num_views = 4

    def __call__(self, features, cls_scores, pids):
        size = features.size(0)
        prob = torch.log_softmax(cls_scores, dim=1)
        probs = prob[torch.arange(size), pids]
        weights = torch.softmax(probs.view(-1, self.num_views), dim=1).view(-1).clone().detach()
        quantified_features = weights.unsqueeze(1) * features  #  注意：调整weight的维度
        return quantified_features
