import torch
import torch.nn as nn

from . import innovation, module
from .backbone import Backbone


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()

        # ------------- Backbone -----------------------
        self.backbone = Backbone()

        # ------------- Hard content branch -----------------------
        # Global
        HARD_FEATURES_DIM = 2048
        HARD_EMBEDDING_DIM = 512
        self.hard_global_embedding = module.embedding.Embedding(HARD_FEATURES_DIM, HARD_EMBEDDING_DIM)

        self.hard_global_head = nn.Sequential(
            *[
                module.GeneralizedMeanPoolingP(),
                module.Classifier(HARD_EMBEDDING_DIM, config.pid_num),
            ]
        )

        # Parts
        PART_NUM = 2
        hard_part_embedding = nn.ModuleList()
        for i in range(PART_NUM):
            hard_part_embedding.append(module.embedding.Embedding(HARD_FEATURES_DIM, HARD_EMBEDDING_DIM))
        self.hard_part_embedding = hard_part_embedding

        hard_part_head = nn.ModuleList()
        for i in range(PART_NUM):
            hard_part_head_item = nn.Sequential(
                *[
                    module.GeneralizedMeanPoolingP(),
                    module.Classifier(HARD_EMBEDDING_DIM, config.pid_num),
                ]
            )
            hard_part_head.append(hard_part_head_item)
        self.hard_part_head = hard_part_head

    def heatmap(self, x):
        return None

    def forward(self, x):
        if self.training:
            hard_features, soft_features_l3, soft_features_l4 = self.backbone(x)
            return hard_features, soft_features_l3, soft_features_l4
        else:
            hard_features, soft_features_l3, soft_features_l4 = self.backbone(x)
            hard_global_embedding = self.hard_global_embedding(hard_features)
            hard_global_bn_features, hard_global_cls_score = self.hard_global_head(hard_global_embedding)
            return hard_global_bn_features
