import torch
from torch.nn import functional as F


class ContrastLoss:
    """
    ContrastLoss 类用于计算特征之间的对比损失。

    参数:
        config: 模型配置参数，用于设置损失计算中的相关参数。
    """

    def __init__(self, config):
        super(ContrastLoss, self).__init__()
        self.config = config

    def __call__(self, features_1, features_2, pids):

        # Method 1 ---------------------------
        bs = features_1.size(0)
        # new_features_2 = torch.zeros(features_1.size()).to(features_1.device)
        # for i in range(int(features_2.size(0))):
        #     new_features_2[i * 4 : i * 4 + 4] = features_2[i]
        new_features_2 = torch.repeat_interleave(new_features_2, 4, dim=0)
        loss = 0.5 * (1 / bs) * torch.norm((features_1 - new_features_2), p=2)
        return loss
