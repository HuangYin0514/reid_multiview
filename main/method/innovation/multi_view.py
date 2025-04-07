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
        C = features.size(1)

        # Reshape: [batch_size, C] -> [chunk_size, view_num, C]
        fused = features.view(chunk_size, self.view_num, C)
        fusion_features = fused.sum(dim=1)  # or .mean(dim=1)

        # 保留每组的第一个pid [size] -> [chunk_size]
        fusion_pids = pids.view(chunk_size, self.view_num)[:, 0]

        return fusion_features, fusion_pids
