import torch
import torch.nn as nn

from .backbone import Backbone
from .net_module import (
    Classifier,
    FeatureDecoupling,
    FeatureReconstruction,
    FeatureVectorIntegrationNet,
    GeneralizedMeanPoolingP,
)


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()

        ####################################
        # IDE
        self.backbone = Backbone()

        ####################################
        # Classifer [bn -> classifier]
        self.backbone_gap = GeneralizedMeanPoolingP()
        self.backbone_classifier = Classifier(2048, config.pid_num)

        self.intergarte_gap = GeneralizedMeanPoolingP()
        self.intergarte_classifier = Classifier(2048, config.pid_num)

        ####################################
        # 解耦
        self.featureDecoupling = FeatureDecoupling(config)
        self.featureVectorIntegrationNet = FeatureVectorIntegrationNet(config)

    def heatmap(self, x):
        _, _, _, _, features_map = self.backbone(x)
        return features_map

    def forward(self, x):
        if self.training:
            x1, x2, x3, x4, features_map = self.backbone(x)
            return features_map
        else:
            ###############
            x1, x2, x3, x4, features_map = self.backbone(x)
            backbone_features = self.backbone_gap(features_map).squeeze()
            backbone_bn_features, backbone_cls_score = self.backbone_classifier(backbone_features)
            return backbone_bn_features
