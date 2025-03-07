import torch
from torch.nn import functional as F


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


class FeatureIntegration:
    """
    该类用于特征整合，将输入的特征按照一定规则整合成更紧凑的特征，并对应地选取pids。

    算法说明:
    - 将输入的features沿batch维度按照chunks进行划分，每个chunk包含4个连续的特征。
    - 对于每个chunk，将这4个特征求和作为整合后的特征。
    - 同时，选取该chunk中第一个特征对应的pids作为整合后的pids。
    """

    def __init__(self, config):
        super(FeatureIntegration, self).__init__()
        self.config = config

    def __call__(self, features, pids):
        size = features.size(0)
        chunk_size = int(size / 4)  # 16
        c = features.size(1)

        integrating_features = torch.zeros([chunk_size, c]).to(features.device)
        integrating_pids = torch.zeros([chunk_size]).to(pids.device)

        for i in range(chunk_size):
            integrating_features[i] = 1 * (features[4 * i] + features[4 * i + 1] + features[4 * i + 2] + features[4 * i + 3])
            integrating_pids[i] = pids[4 * i]

        return integrating_features, integrating_pids


class ContrastLoss:

    def __init__(self, config):
        super(ContrastLoss, self).__init__()
        self.config = config

    def __call__(self, features_1, features_2):
        new_features_2 = torch.repeat_interleave(features_2, repeats=4, dim=0).clone().detach()
        loss = F.normalize(features_1 - new_features_2, p=2, dim=1).mean(0).sum()
        return loss
