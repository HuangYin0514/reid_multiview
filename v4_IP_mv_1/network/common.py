import torch
import torch.nn as nn

from .gem_pool import GeneralizedMeanPoolingP


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_out")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif classname.find("Conv") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find("BatchNorm") != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
    elif classname.find("InstanceNorm") != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


class GAP_BN(nn.Module):
    def __init__(self, channel=2048):
        super(GAP_BN, self).__init__()
        self.GAP = GeneralizedMeanPoolingP()
        # self.GAP = nn.AdaptiveAvgPool2d(1)
        self.BN = nn.BatchNorm1d(channel)
        self.BN.apply(weights_init_kaiming)

    def forward(self, features_map):
        features = self.GAP(features_map)
        bn_features = self.BN(features.squeeze())
        return bn_features
