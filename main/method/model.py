import torch
import torch.nn as nn

from . import innovation, module
from .backbone import Backbone


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        BACKBONE_FEATURES_DIM = config.MODEL.BACKBONE_FEATURES_DIM
        EMBEDDING_FEATURES_DIM = config.MODEL.EMBEDDING_FEATURES_DIM
        PART_NUM = config.MODEL.PART_NUM
        NUM_ATTENTION = config.MODEL.NUM_ATTENTION

        # ------------- Backbone -----------------------
        self.backbone = Backbone(config)

        # ------------- Hard content branch -----------------------
        # Global
        self.hard_global_embedding = module.embedding.Embedding(BACKBONE_FEATURES_DIM, EMBEDDING_FEATURES_DIM)
        self.hard_global_pooling = module.GeneralizedMeanPoolingP()
        self.hard_global_classifier = module.Classifier(EMBEDDING_FEATURES_DIM, config.DATASET.PID_NUM)

        # Parts
        hard_part_embedding = nn.ModuleList()
        hard_part_pooling = nn.ModuleList()
        hard_part_classifier = nn.ModuleList()
        for i in range(PART_NUM):
            hard_part_embedding.append(module.embedding.Embedding(BACKBONE_FEATURES_DIM, EMBEDDING_FEATURES_DIM))
            hard_part_pooling.append(module.GeneralizedMeanPoolingP())
            hard_part_classifier.append(module.Classifier(EMBEDDING_FEATURES_DIM, config.DATASET.PID_NUM))
        self.hard_part_embedding = hard_part_embedding
        self.hard_part_pooling = hard_part_pooling
        self.hard_part_classifier = hard_part_classifier

        # ------------- Multiview content branch -----------------------
        self.multiview_global_embedding = module.embedding.Embedding(BACKBONE_FEATURES_DIM, EMBEDDING_FEATURES_DIM)
        self.multiview_global_pooling = module.GeneralizedMeanPoolingP()
        self.multiview_global_decoupling = innovation.decoupling.Feature_Decoupling_Net(EMBEDDING_FEATURES_DIM, EMBEDDING_FEATURES_DIM)
        self.multiview_global_shared_feature_fusion = innovation.decoupling.Feature_Fusion_Net(EMBEDDING_FEATURES_DIM, EMBEDDING_FEATURES_DIM, config.MODEL.VIEW_NUM)
        self.multiview_global_specific_feature_fusion = innovation.decoupling.Feature_Fusion_Net(EMBEDDING_FEATURES_DIM, EMBEDDING_FEATURES_DIM, config.MODEL.VIEW_NUM)
        self.multiview_global_classifier = module.Classifier(EMBEDDING_FEATURES_DIM * 2, config.DATASET.PID_NUM)

    def heatmap(self, x):
        return None

    def forward(self, x):
        if self.training:
            hard_features, soft_features_l3, soft_features_l4 = self.backbone(x)
            return hard_features, soft_features_l3, soft_features_l4
        else:
            eval_features = []
            hard_features, soft_features_l3, soft_features_l4 = self.backbone(x)

            # ------------- Hard content branch -----------------------
            ## Global
            hard_global_embedding_features = self.hard_global_embedding(hard_features)
            hard_global_pooling_features = self.hard_global_pooling(hard_global_embedding_features).squeeze()
            hard_global_bn_features, hard_global_cls_score = self.hard_global_classifier(hard_global_pooling_features)
            eval_features.append(hard_global_bn_features)

            # ## Parts
            PART_NUM = self.config.MODEL.PART_NUM
            hard_part_chunk_features = torch.chunk(hard_features, PART_NUM, dim=2)
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
