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

    def __init__(self, config):
        super(ContrastLoss, self).__init__()
        self.config = config

    def __call__(self, features_1, features_2):
        new_features_2 = torch.repeat_interleave(features_2, repeats=4, dim=0).clone().detach()
        loss = F.normalize(features_1 - new_features_2, p=2, dim=1).mean(0).sum() + F.normalize(features_1, p=2, dim=1).mean(0).sum()
        return loss
