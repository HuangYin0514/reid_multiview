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
        c = features.size(1)

        fusion_features = torch.zeros([chunk_size, c]).to(features.device)
        fusion_pids = torch.zeros([chunk_size]).to(pids.device)

        for i in range(chunk_size):
            fusion_features[i] = 1 * (features[self.view_num * i] + features[self.view_num * i + 1] + features[self.view_num * i + 2] + features[self.view_num * i + 3])
            fusion_pids[i] = pids[self.view_num * i]

        return fusion_features, fusion_pids
