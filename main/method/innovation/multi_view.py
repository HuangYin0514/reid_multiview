import torch
from torch.nn import functional as F


class FeatureIntegration:
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
        new_features_2 = torch.repeat_interleave(features_2, repeats=4, dim=0).clone().detach()
        loss = torch.norm((features_1 - new_features_2), p=2)
        return loss

    def v1(self, features_1, features_2):
        # v463
        new_features_2 = torch.zeros(features_1.size()).to(features_1.device)
        for i in range(int(new_features_2.size(0) / 4)):
            new_features_2[i * 4 : i * 4 + 4] = features_2[i]
        loss = torch.norm((features_1 - new_features_2), p=2)
        return loss

    def v2(self, features_1, features_2):
        # v462
        loss = torch.norm(features_1, p=2)
        return loss
