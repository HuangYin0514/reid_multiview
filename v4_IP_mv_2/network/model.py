import torch
import torch.nn as nn
from tools import CrossEntropyLabelSmooth

from .common import *
from .contrastive_loss import *


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.backbone = Backbone()

        # 分类
        self.gap_bn = GAP_BN(2048)
        self.bn_classifier = BN_Classifier(2048, config.pid_num)
        self.bn_classifier2 = BN_Classifier(2048, config.pid_num)

        # 解耦
        self.decoupling = FeatureDecoupling(config)
        self.decoupling_shared_bn_classifier = BN_Classifier(1024, config.pid_num)
        self.decoupling_special_bn_classifier = BN_Classifier(1024, config.pid_num)

        # 特征融合
        self.feature_integrating = FeatureMapIntegrating(config)

    def heatmap(self, x):
        _, _, _, _, features_map = self.backbone(x)
        return features_map

    def forward(self, x):
        if self.training:
            _, _, _, _, features_map = self.backbone(x)
            return features_map
        else:
            _, _, _, _, features_map = self.backbone(x)
            bn_features = self.gap_bn(features_map)
            shared_features, special_features = self.decoupling(bn_features)
            bn_features = torch.cat([shared_features, special_features], dim=1)

            flip_x = torch.flip(x, [3])
            _, _, _, _, flip_features_map = self.backbone(flip_x)
            flip_bn_features = self.gap_bn(flip_features_map)
            flip_shared_features, flip_special_features = self.decoupling(flip_bn_features)
            flip_bn_features = torch.cat([flip_shared_features, flip_special_features], dim=1)

            bn_features = bn_features + flip_bn_features
            return bn_features
