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
        self.backbone_classifier = Classifier(768, config.pid_num)

    def forward(self, x, cids):
        if self.training:
            features = self.backbone(x, cids)
            return features
        else:
            ###############
            features = self.backbone(x, cids)
            backbone_bn_features, backbone_cls_score = self.backbone_classifier(features)
            return backbone_bn_features
