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
        self.hard_global_pooling = module.GeneralizedMeanPoolingP()
        self.hard_global_classifier = module.Classifier(HARD_EMBEDDING_DIM, config.pid_num)

        # Parts
        PART_NUM = 2
        hard_part_embedding = nn.ModuleList()
        hard_part_pooling = nn.ModuleList()
        hard_part_classifier = nn.ModuleList()
        for i in range(PART_NUM):
            hard_part_embedding.append(module.embedding.Embedding(HARD_FEATURES_DIM, HARD_EMBEDDING_DIM))
            hard_part_pooling.append(module.GeneralizedMeanPoolingP())
            hard_part_classifier.append(module.Classifier(HARD_EMBEDDING_DIM, config.pid_num))
        self.hard_part_embedding = hard_part_embedding
        self.hard_part_pooling = hard_part_pooling
        self.hard_part_classifier = hard_part_classifier

        # ------------- soft content branch -----------------------
        # Downstream
        SOFT_FEATURES_DIM = 2048
        SOFT_EMBEDDING_DIM = 512
        self.soft_downstream_l4_embedding = module.embedding.Embedding(SOFT_FEATURES_DIM // 2, SOFT_FEATURES_DIM // 2)
        self.soft_downstream_global_embedding = module.embedding.Embedding(SOFT_FEATURES_DIM // 2, SOFT_EMBEDDING_DIM)
        self.soft_downstream_global_pooling = module.GeneralizedMeanPoolingP()
        self.soft_downstream_global_classifier = module.Classifier(SOFT_EMBEDDING_DIM, config.pid_num)

        self.soft_downstream_attention = None
        self.soft_downstream_attention_classifier = None

        # Upstream
        self.soft_upstream_global_embedding = module.embedding.Embedding(SOFT_FEATURES_DIM, SOFT_EMBEDDING_DIM)
        self.soft_upstream_global_pooling = module.GeneralizedMeanPoolingP()
        self.soft_upstream_global_classifier = module.Classifier(SOFT_EMBEDDING_DIM, config.pid_num)

        self.soft_upstream_attention = None
        self.soft_upstream_attention_classifier = None

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
            # PART_NUM = 2
            # hard_part_chunk_features = torch.chunk(hard_features, PART_NUM, dim=2)
            # for i in range(PART_NUM):
            #     hard_part_chunk_feature_item = hard_part_chunk_features[i]
            #     hard_part_embedding_features = self.hard_part_embedding[i](hard_part_chunk_feature_item)
            #     hard_part_pooling_features = self.hard_part_pooling[i](hard_part_embedding_features).squeeze()
            #     hard_part_bn_features, hard_part_cls_score = self.hard_part_classifier[i](hard_part_pooling_features)
            #     eval_features.append(hard_part_bn_features)

            eval_features = torch.cat(eval_features, dim=1)

            return eval_features
