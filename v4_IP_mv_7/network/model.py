import torch
import torch.nn as nn
from tools import CrossEntropyLabelSmooth

from .common import *
from .contrastive_loss import *


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.backbone = Backbone()

        # 解耦
        self.decoupling = FeatureDecoupling(config)
        self.decoupling_shared_bn_classifier = BN_Classifier(1024, config.pid_num)
        self.decoupling_special_bn_classifier = BN_Classifier(1024, config.pid_num)

        # 特征融合
        self.feature_integrating = FeatureMapIntegrating(config)

        # 分类
        self.gap_bn = GAP_BN(2048)
        self.bn_classifier = BN_Classifier(2048, config.pid_num)
        self.bn_classifier2 = BN_Classifier(2048, config.pid_num)

    def heatmap(self, x):
        _, _, _, _, features_map = self.backbone(x)
        return features_map

    def forward(self, x):
        if self.training:
            x1, x2, x3, x4, backbone_features_map = self.backbone(x)
            bn_features = self.gap_bn(backbone_features_map)
            return bn_features
        else:

            def extract_features(self, x):
                x1, x2, x3, x4, backbone_features_map = self.backbone(x)
                bn_features = self.gap_bn(backbone_features_map)
                return bn_features

            bn_features = extract_features(x)
            flip_images = torch.flip(x, [3])
            flip_bn_features = extract_features(flip_images)
            bn_features = bn_features + flip_bn_features
            return bn_features
