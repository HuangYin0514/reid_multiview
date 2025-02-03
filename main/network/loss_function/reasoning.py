import torch
import torch.nn as nn
import torch.nn.functional as F


class ReasoningLoss(nn.Module):
    def __init__(self):
        super(ReasoningLoss, self).__init__()

    def forward(self, bn_features, bn_features2):
        new_bn_features2 = torch.zeros(bn_features.size()).to(bn_features.device)
        for i in range(int(bn_features2.size(0) / 4)):
            new_bn_features2[i * 4 : i * 4 + 4] = bn_features2[i]
        loss = torch.norm((bn_features - new_bn_features2), p=2)
        return loss
