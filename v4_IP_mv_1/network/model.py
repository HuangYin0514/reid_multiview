import torch
import torch.nn as nn

from .common import *


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.backbone = Backbone()

        ###########################################################
        # 分类
        self.gap_bn = GAP_BN(2048)
        self.bn_classifier = BN_Classifier(2048, config.pid_num)

        self.gap_bn2 = GAP_BN(2048)
        self.bn_classifier2 = BN_Classifier(2048, config.pid_num)

        self.gap_bn3 = GAP_BN(2048)
        self.bn_classifier3 = BN_Classifier(2048, config.pid_num)

        ###########################################################
        # 解耦
        self.decoupling = FeatureDecoupling(config)
        self.decoupling_reconstruction = FeatureDecouplingReconstruction(config)

        self.decoupling_gap_bn = GAP_BN(2048)

        self.decoupling_shared_gap_bn = GAP_BN(2048)
        self.decoupling_shared_bn_classifier = BN_Classifier(1024, config.pid_num)

        self.decoupling_special_gap_bn = GAP_BN(2048)
        self.decoupling_special_bn_classifier = BN_Classifier(1024, config.pid_num)

        ###########################################################
        # 多视角特征聚合
        self.feature_integrating = FeatureIntegrating(config)

        self.quantified_gap_bn = GAP_BN(2048)

        ###########################################################
        # 特征融合
        self.feature_fusion = FeatureFusion(config)

    def heatmap(self, x):
        _, _, _, _, features_map = self.backbone(x)
        return features_map

    def forward(self, x):
        if self.training:
            x1, x2, x3, x4, features_map = self.backbone(x)
            return features_map
        else:
            x1, x2, x3, x4, backbone_features_map = self.backbone(x)
            bn_features = self.gap_bn(backbone_features_map)
            # shared_features, special_features = self.decoupling(backbone_features)
            # bn_features = self.feature_fusion(shared_features, special_features)
            bn_features, cls_score = self.bn_classifier(bn_features)
            return bn_features
