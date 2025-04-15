import torch
import torch.nn as nn

from . import innovation, module
from .backbone import Backbone


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        BACKBONE_FEATURES_DIM = config.MODEL.BACKBONE_FEATURES_DIM
        VIEW_NUM = config.MODEL.VIEW_NUM
        PID_NUM = config.DATASET.PID_NUM
        PART_NUM = config.MODEL.PART_NUM
        EMBEDDING_FEATURES_DIM = config.MODEL.EMBEDDING_FEATURES_DIM

        # ------------- Backbone -----------------------
        self.backbone = Backbone()

        # ------------- Hard content branch -----------------------
        # Global
        self.global_pooling = module.GeneralizedMeanPoolingP()
        self.global_classifier = module.Classifier(BACKBONE_FEATURES_DIM, PID_NUM)

        # Part
        self.hard_part_pooling = nn.ModuleList()
        self.hard_part_embedding = nn.ModuleList()
        self.hard_part_classifier = nn.ModuleList()
        for i in range(PART_NUM):
            self.hard_part_pooling.append(module.GeneralizedMeanPoolingP())
            self.hard_part_embedding.append(module.embedding.FCEmbedding(BACKBONE_FEATURES_DIM, EMBEDDING_FEATURES_DIM))
            self.hard_part_classifier.append(module.Classifier(EMBEDDING_FEATURES_DIM, PID_NUM))

        # ------------- Multiview content branch -----------------------
        self.multiview_pooling = module.GeneralizedMeanPoolingP()
        self.multiview_feature_map_location = innovation.multi_view.FeatureMapLocation()
        self.multiview_feature_quantification = innovation.multi_view.FeatureQuantification(VIEW_NUM)
        self.multiview_feature_fusion = innovation.multi_view.MultiviewFeatureFusion(VIEW_NUM)
        self.multiview_classifier = module.Classifier(BACKBONE_FEATURES_DIM, PID_NUM)

        # ------------- Contrast  Module -----------------------
        self.contrast_kl_loss = innovation.multi_view.MVDistillKL(VIEW_NUM)

    def heatmap(self, x):
        return None

    def forward(self, x):
        if self.training:
            resnet_feature_maps = self.backbone(x)
            return resnet_feature_maps
        else:
            eval_features = []
            resnet_feature_maps = self.backbone(x)

            global_features = self.global_pooling(resnet_feature_maps).squeeze()
            global_bn_features, global_cls_score = self.global_classifier(global_features)
            eval_features.append(global_bn_features)

            eval_features = torch.cat(eval_features, dim=1)

            return eval_features
