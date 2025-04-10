import torch
import torch.nn as nn

from . import innovation, module
from .backbone import Backbone


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()

        # ------------- Parameter -----------------------
        self.config = config
        BACKBONE_FEATURES_DIM = config.MODEL.BACKBONE_FEATURES_DIM
        VIEW_NUM = config.MODEL.VIEW_NUM

        PID_NUM = config.DATASET.PID_NUM

        EMBEDDING_FEATURES_DIM = config.MODEL.EMBEDDING_FEATURES_DIM
        PART_NUM = config.MODEL.PART_NUM

        # ------------- Backbone -----------------------
        self.backbone = Backbone()

        # ------------- Hard content branch -----------------------
        # Global
        self.hard_global_embedding = module.embedding.Embedding(BACKBONE_FEATURES_DIM, EMBEDDING_FEATURES_DIM)
        self.hard_global_pooling = module.GeneralizedMeanPoolingP()
        self.hard_global_classifier = module.Classifier(EMBEDDING_FEATURES_DIM, PID_NUM)

        # Parts
        hard_part_embedding = nn.ModuleList()
        hard_part_pooling = nn.ModuleList()
        hard_part_classifier = nn.ModuleList()
        for i in range(PART_NUM):
            hard_part_embedding.append(module.embedding.Embedding(BACKBONE_FEATURES_DIM, EMBEDDING_FEATURES_DIM))
            hard_part_pooling.append(module.GeneralizedMeanPoolingP())
            hard_part_classifier.append(module.Classifier(EMBEDDING_FEATURES_DIM, PID_NUM))
        self.hard_part_embedding = hard_part_embedding
        self.hard_part_pooling = hard_part_pooling
        self.hard_part_classifier = hard_part_classifier

        # ------------- Multiview content branch -----------------------
        self.multiview_pooling = module.GeneralizedMeanPoolingP()
        self.multiview_feature_map_location = innovation.multi_view.FeatureMapLocation()
        self.multiview_feature_quantification = innovation.multi_view.FeatureQuantification(VIEW_NUM)
        self.multiview_feature_fusion = innovation.multi_view.MultiviewFeatureFusion(VIEW_NUM)
        self.multiview_classifier = module.Classifier(EMBEDDING_FEATURES_DIM, PID_NUM)
        self.contrast_loss = innovation.multi_view.ContrastLoss(VIEW_NUM)

    def heatmap(self, x):
        return None

    def forward(self, x):
        if self.training:
            resnet_feature_maps = self.backbone(x)
            return resnet_feature_maps
        else:
            eval_features = []
            resnet_feature_maps = self.backbone(x)

            # ------------- Hard content branch -----------------------
            # Global
            hard_global_embedding_features = self.hard_global_embedding(resnet_feature_maps)
            hard_global_pooling_features = self.hard_global_pooling(hard_global_embedding_features).squeeze()
            hard_global_bn_features, hard_global_cls_score = self.hard_global_classifier(hard_global_pooling_features)
            eval_features.append(hard_global_bn_features)

            # Parts
            PART_NUM = self.config.MODEL.PART_NUM
            hard_part_chunk_features = torch.chunk(resnet_feature_maps, PART_NUM, dim=2)
            hard_part_embedding_features_list = []
            for i in range(PART_NUM):
                hard_part_chunk_feature_item = hard_part_chunk_features[i]
                hard_part_embedding_features = self.hard_part_embedding[i](hard_part_chunk_feature_item)
                hard_part_embedding_features_list.append(hard_part_embedding_features)
                hard_part_pooling_features = self.hard_part_pooling[i](hard_part_embedding_features).squeeze()
                hard_part_bn_features, hard_part_cls_score = self.hard_part_classifier[i](hard_part_pooling_features)
                eval_features.append(hard_part_bn_features)

            eval_features = torch.cat(eval_features, dim=1)

            return eval_features
