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

        # ------------- soft content branch -----------------------
        # Upstream
        self.soft_upstream_global_embedding = module.embedding.Embedding(BACKBONE_FEATURES_DIM, EMBEDDING_FEATURES_DIM)
        self.soft_upstream_global_pooling = module.GeneralizedMeanPoolingP()
        self.soft_upstream_global_classifier = module.Classifier(EMBEDDING_FEATURES_DIM, config.DATASET.PID_NUM)

        self.soft_upstream_attention = innovation.dualscale_attention.Dualscale_Attention(BACKBONE_FEATURES_DIM, EMBEDDING_FEATURES_DIM, NUM_ATTENTION)
        self.soft_upstream_attention_classifier = module.Classifier(EMBEDDING_FEATURES_DIM * NUM_ATTENTION, config.DATASET.PID_NUM)

        # Downstream
        self.soft_downstream_l4_embedding = module.embedding.Embedding(BACKBONE_FEATURES_DIM // 2, BACKBONE_FEATURES_DIM // 2)
        self.soft_downstream_global_embedding = module.embedding.Embedding(BACKBONE_FEATURES_DIM // 2, EMBEDDING_FEATURES_DIM)
        self.soft_downstream_global_pooling = module.GeneralizedMeanPoolingP()
        self.soft_downstream_global_classifier = module.Classifier(EMBEDDING_FEATURES_DIM, config.DATASET.PID_NUM)

        self.guide_dualscale_attention = innovation.dualscale_attention.Guide_Dualscale_Attention(BACKBONE_FEATURES_DIM // 2, EMBEDDING_FEATURES_DIM, NUM_ATTENTION)
        self.soft_downstream_attention_classifier = module.Classifier(EMBEDDING_FEATURES_DIM * NUM_ATTENTION, config.DATASET.PID_NUM)

        # ------------- fuson content branch -----------------------
        self.fusion = innovation.fusion.Fusion(EMBEDDING_FEATURES_DIM)
        self.fusion_pooling = module.GeneralizedMeanPoolingP()
        self.fusion_classifier = module.Classifier(EMBEDDING_FEATURES_DIM * 4, config.DATASET.PID_NUM)

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

            # ------------- Soft content branch -----------------------
            # Upstream
            soft_upstream_global_features = self.soft_upstream_global_embedding(soft_features_l4)
            soft_upstream_global_pooling_features = self.soft_upstream_global_pooling(soft_upstream_global_features).squeeze()
            soft_upstream_global_bn_features, soft_upstream_global_cls_score = self.soft_upstream_global_classifier(soft_upstream_global_pooling_features)
            eval_features.append(soft_upstream_global_bn_features)

            # Upstream attention
            soft_upstream_attention_attentions, soft_upstream_attention_bap_AiF_features, soft_upstream_attention_bap_features = self.soft_upstream_attention(soft_features_l4)
            soft_upstream_attention_bn_features, soft_upstream_attention_cls_score = self.soft_upstream_attention_classifier(soft_upstream_attention_bap_features)
            eval_features.append(soft_upstream_attention_bn_features)

            # Downstream
            soft_downstream_l4_embedding_features = self.soft_downstream_l4_embedding(soft_features_l3)
            soft_downstream_global_embedding_features = self.soft_downstream_global_embedding(soft_downstream_l4_embedding_features)
            soft_downstream_global_pooling_features = self.soft_downstream_global_pooling(soft_downstream_global_embedding_features).squeeze()
            soft_downstream_global_bn_features, soft_downstream_global_cls_score = self.soft_downstream_global_classifier(soft_downstream_global_pooling_features)
            eval_features.append(soft_downstream_global_bn_features)

            # Downstream attention
            soft_downstream_attention_attentions, soft_downstream_attention_bap_AiF_features, soft_downstream_attention_bap_features = self.guide_dualscale_attention(soft_features_l3, soft_downstream_l4_embedding_features, soft_upstream_attention_attentions)
            soft_downstream_attention_bn_features, soft_downstream_attention_cls_score = self.soft_downstream_attention_classifier(soft_downstream_attention_bap_features)
            eval_features.append(soft_downstream_attention_bn_features)

            # # ------------- Fusion content branch -----------------------
            # fusion_features = self.fusion(soft_upstream_global_features, hard_part_embedding_features_list, soft_upstream_global_features, soft_downstream_global_embedding_features)
            # fusion_pooling_features = self.fusion_pooling(fusion_features).squeeze()
            # fusion_bn_features, fusion_cls_score = self.fusion_classifier(fusion_pooling_features)
            # eval_features.append(fusion_bn_features)

            eval_features = torch.cat(eval_features, dim=1)

            return eval_features
