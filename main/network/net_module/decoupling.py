import torch
import torch.nn as nn

from .weights_init import weights_init_classifier, weights_init_kaiming


class FeatureDecoupling(nn.Module):
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
        super(FeatureDecoupling, self).__init__()
        self.config = config

        # shared branch
        ic = 2048
        oc = 1024
        self.mlp1 = nn.Sequential(
            nn.Linear(ic, oc, bias=False),
            nn.BatchNorm1d(oc),
            nn.ReLU(),
            nn.Linear(oc, oc, bias=False),
            nn.BatchNorm1d(oc),
        )
        self.mlp1.apply(weights_init_kaiming)

        # special branch
        self.mlp2 = nn.Sequential(
            nn.Linear(ic, oc, bias=False),
            nn.BatchNorm1d(oc),
            nn.ReLU(),
            nn.Linear(oc, oc, bias=False),
            nn.BatchNorm1d(oc),
        )
        self.mlp2.apply(weights_init_kaiming)

    def forward(self, features):
        shared_features = self.mlp1(features)
        special_features = self.mlp2(features)
        return shared_features, special_features


class FeatureReconstruction(nn.Module):
    def __init__(self, config):
        super(FeatureReconstruction, self).__init__()
        self.config = config

    def __call__(self, features_1, features_2):
        out = torch.cat([features_1, features_2], dim=1)
        return out


class FeatureVectorIntegrationNet(nn.Module):
    def __init__(self, config):
        super(FeatureVectorIntegrationNet, self).__init__()
        self.config = config

        # shared branch
        ic = 1024 * 4
        oc = 1024
        self.mlp1 = nn.Sequential(
            nn.Linear(ic, oc, bias=False),
            nn.BatchNorm1d(oc),
        )
        self.mlp1.apply(weights_init_kaiming)

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
