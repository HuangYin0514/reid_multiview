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

        # ------------- Backbone -----------------------
        self.backbone = Backbone()

        # ------------- Global content branch -----------------------
        # Global
        self.global_pooling = module.GeneralizedMeanPoolingP()
        self.global_classifier = module.Classifier(BACKBONE_FEATURES_DIM, PID_NUM)

        # ------------- Multiview content branch -----------------------
        self.multiview_pooling = module.GeneralizedMeanPoolingP()
        self.multiview_feature_map_location = innovation.multi_view.FeatureMapLocation()
        self.multiview_feature_quantification = innovation.multi_view.FeatureQuantification(VIEW_NUM)
        self.multiview_feature_fusion = innovation.multi_view.MultiviewFeatureFusion(VIEW_NUM)
        self.multiview_classifier = module.Classifier(BACKBONE_FEATURES_DIM, PID_NUM)
        self.contrast_loss = innovation.multi_view.ContrastLoss(VIEW_NUM)

        # ------------- Hard content branch -----------------------
        # Part
        hard_part_pooling = nn.ModuleList()
        hard_part_classifier = nn.ModuleList()
        for i in range(PART_NUM):
            hard_part_pooling.append(module.GeneralizedMeanPoolingP())
            hard_part_classifier.append(module.Classifier(BACKBONE_FEATURES_DIM, PID_NUM))
        self.hard_part_pooling = hard_part_pooling
        self.hard_part_classifier = hard_part_classifier

    def heatmap(self, x):
        return None

    def forward(self, x):
        if self.training:
            resnet_feature_maps = self.backbone(x)
            return resnet_feature_maps
        else:
            eval_features = []
            resnet_feature_maps = self.backbone(x)

            # ------------- Global content branch -----------------------
            # Global
            global_features = self.global_pooling(resnet_feature_maps).squeeze()
            global_bn_features, global_cls_score = self.global_classifier(global_features)
            eval_features.append(global_bn_features)

            # # Parts
            # PART_NUM = self.config.MODEL.PART_NUM
            # hard_part_chunk_features = torch.chunk(resnet_feature_maps, PART_NUM, dim=2)
            # for i in range(PART_NUM):
            #     hard_part_chunk_feature_item = hard_part_chunk_features[i]
            #     hard_part_pooling_features = self.hard_part_pooling[i](hard_part_chunk_feature_item).squeeze()
            #     hard_part_bn_features, hard_part_cls_score = self.hard_part_classifier[i](hard_part_pooling_features)
            #     eval_features.append(hard_part_bn_features)

            eval_features = torch.cat(eval_features, dim=1)

            return eval_features
