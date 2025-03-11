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


# class ContrastLoss:

#     def __init__(self, config):
#         super(ContrastLoss, self).__init__()
#         self.config = config

#     def __call__(self, features_1, features_2):
#         loss = torch.norm(features_1, p=2)
#         return loss


# class ContrastLoss:

#     def __init__(self, config):
#         super(ContrastLoss, self).__init__()
#         self.config = config

#     def __call__(self, features_1, features_2):
#         # new_features_2 = torch.repeat_interleave(features_2, repeats=4, dim=0).clone().detach()
#         new_features_2 = torch.zeros(features_1.size()).to(features_1.device)
#         for i in range(int(new_features_2.size(0) / 4)):
#             new_features_2[i * 4 : i * 4 + 4] = features_2[i]
#         loss = torch.norm((features_1 - new_features_2), p=2)
#         return loss


class ContrastLoss:

    def __init__(self, config):
        super(ContrastLoss, self).__init__()
        self.config = config

    def __call__(self, features_1, features_2):
        new_features_2 = torch.repeat_interleave(features_2, repeats=4, dim=0).clone().detach()
        loss = torch.norm((features_1 - new_features_2), p=2)
        return loss
