import copy
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .. import module


class Fusion(nn.Module):
    def __init__(self, feat_dim) -> None:
        super(Fusion, self).__init__()

        self.merge = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, 1, 1, bias=False),
            nn.Conv2d(feat_dim, feat_dim, 3, 1, 1, bias=False),
            module.get_norm("BN", feat_dim),
            nn.ReLU(),
        )
        self.merge.apply(module.weights_init_kaiming)

        self.output_part = nn.Sequential(
            nn.Conv2d(feat_dim * 2, feat_dim * 2, 1, 1, bias=False),
            nn.Conv2d(feat_dim * 2, feat_dim * 2, 3, 1, 1, bias=False),
            module.get_norm("BN", feat_dim * 2),
            nn.ReLU(),
        )
        self.output_part.apply(module.weights_init_kaiming)

        self.output_attn = nn.Sequential(
            nn.Conv2d(feat_dim * 2, feat_dim * 2, 1, 1, bias=False),
            nn.Conv2d(feat_dim * 2, feat_dim * 2, 3, 1, 1, bias=False),
            module.get_norm("BN", feat_dim * 2),
            nn.ReLU(),
        )
        self.output_attn.apply(module.weights_init_kaiming)

        self.output = nn.Sequential(
            nn.Conv2d(feat_dim * 4, feat_dim * 4, 1, 1, bias=False),
            nn.Conv2d(feat_dim * 4, feat_dim * 4, 3, 1, 1, bias=False),
            module.get_norm("BN", feat_dim * 4),
            nn.ReLU(),
        )
        self.output.apply(module.weights_init_kaiming)

    def forward(self, hard_global, hard_part, attn_feature4, attn_feature3):
        hard_part_merge = self.merge(torch.cat(hard_part, dim=2))
        part_merge = self.output_part(torch.cat([hard_part_merge, hard_global], dim=1))
        attn_merge = self.output_attn(torch.cat([attn_feature4, attn_feature3], dim=1))
        return self.output(torch.cat([part_merge, attn_merge], dim=1))
