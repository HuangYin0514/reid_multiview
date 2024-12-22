import torch
import torch.nn as nn

from .common import *


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.backbone = Backbone()

        # 解耦
        self.decoupling = FeatureDecoupling(config)
        self.decoupling_reconstruction = FeatureDecouplingReconstruction(config)
        self.decoupling_shared_bn_classifier = BN_Classifier(1024, config.pid_num)
        self.decoupling_special_bn_classifier = BN_Classifier(1024, config.pid_num)

        # 多视角特征聚合
        self.feature_integrating = FeatureIntegrating(config)

        # 特征融合
        self.feature_fusion = FeatureFusion(config)

        # 分类
        self.gap_bn = GAP_BN(2048)
        self.bn_classifier = BN_Classifier(2048, config.pid_num)
        self.bn_classifier2 = BN_Classifier(2048, config.pid_num)

    def heatmap(self, x):
        _, _, _, _, features_map = self.backbone(x)
        return features_map

    def forward(self, x):
        if self.training:
            x1, x2, x3, x4, features_map = self.backbone(x)
            return features_map
        else:
            x1, x2, x3, x4, backbone_features_map = self.backbone(x)
            backbone_features = self.gap_bn(backbone_features_map)
            shared_features, special_features = self.decoupling(backbone_features)
            features = self.feature_fusion(shared_features, special_features)
            bn_features, cls_score = self.bn_classifier(features)
            return bn_features
