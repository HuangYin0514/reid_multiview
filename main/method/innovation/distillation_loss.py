import copy
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class Distillation_Loss(nn.Module):
    def __init__(self, feature_dim):
        super(Distillation_Loss, self).__init__()
        self.predictor1 = nn.Sequential(
            nn.Linear(feature_dim, feature_dim, bias=False),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim),
        )
        self.predictor2 = nn.Sequential(
            nn.Linear(feature_dim, feature_dim, bias=False),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim),
        )

    def forward(self, featrue_1, featrue_2):
        z1 = featrue_1
        z2 = featrue_2
        p1 = self.predictor1(z1)
        p2 = self.predictor2(z2)
        loss = -0.5 * (F.cosine_similarity(p1, z2.detach(), dim=1).mean() + F.cosine_similarity(p2, z1.detach(), dim=1).mean())
        return loss
