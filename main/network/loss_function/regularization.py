import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureRegularizationLoss(nn.Module):
    def __init__(self):
        super(FeatureRegularizationLoss, self).__init__()

    def forward(self, bn_features):
        loss = torch.norm((bn_features), p=2)
        return loss
