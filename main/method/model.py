import torch
import torch.nn as nn

from . import innovation, module
from .backbone import Backbone


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()

        ####################################
        # Backbone
        self.backbone = Backbone()

        ####################################
        # Classifer [bn -> classifier]
        self.backbone_gap = module.GeneralizedMeanPoolingP()
        self.backbone_classifier = module.Classifier(2048, config.pid_num)

        self.intergarte_gap = module.GeneralizedMeanPoolingP()
        self.intergarte_classifier = module.Classifier(2048, config.pid_num)

        ####################################
        # Decoupling & Integration
        self.featureDecouplingModule = innovation.decoupling.FeatureDecouplingModule(config)
        self.featureIntegrationModule = innovation.decoupling.FeatureIntegrationModule(config)

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
