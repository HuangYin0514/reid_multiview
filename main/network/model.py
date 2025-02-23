import torch
import torch.nn as nn

from .backbone import Backbone
from .net_module import (
    Classifier,
    FeatureDecoupling,
    FeatureReconstruction,
    FeatureVectorIntegrationNet,
    GeneralizedMeanPoolingP,
    shuffle_unit,
)


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()

        ####################################
        # IDE
        self.backbone = Backbone()

        ####################################
        # Classifer [bn -> classifier]
        in_planes = 768
        self.backbone_classifier = Classifier(in_planes, config.pid_num)

        ####################################
        # JPM branch
        self.local_classifier_1 = Classifier(in_planes, config.pid_num)
        self.local_classifier_2 = Classifier(in_planes, config.pid_num)
        self.local_classifier_3 = Classifier(in_planes, config.pid_num)
        self.local_classifier_4 = Classifier(in_planes, config.pid_num)
        self.shuffle_groups = 2
        print("using shuffle_groups size:{}".format(self.shuffle_groups))
        self.shift_num = 5
        print("using shift_num size:{}".format(self.shift_num))
        self.divide_length = 4
        print("using divide_length size:{}".format(self.divide_length))

    def forward(self, x):
        if self.training:
            features = self.backbone(x)

            b1_feat = self.backbone.b1(features)  # [64, 129, 768]
            global_feat = b1_feat[:, 0]  # [64, 768]

            # JPM branch
            feature_length = features.size(1) - 1
            patch_length = feature_length // self.divide_length
            token = features[:, 0:1]
            x = shuffle_unit(features, self.shift_num, self.shuffle_groups)

            # Transformer of Last Layer
            # lf_1
            b1_local_feat = x[:, :patch_length]
            b1_local_feat = self.backbone.b2(torch.cat((token, b1_local_feat), dim=1))
            local_feat_1 = b1_local_feat[:, 0]
            # lf_2
            b2_local_feat = x[:, patch_length : patch_length * 2]
            b2_local_feat = self.backbone.b2(torch.cat((token, b2_local_feat), dim=1))
            local_feat_2 = b2_local_feat[:, 0]
            # lf_3
            b3_local_feat = x[:, patch_length * 2 : patch_length * 3]
            b3_local_feat = self.backbone.b2(torch.cat((token, b3_local_feat), dim=1))
            local_feat_3 = b3_local_feat[:, 0]
            # lf_4
            b4_local_feat = x[:, patch_length * 3 : patch_length * 4]
            b4_local_feat = self.backbone.b2(torch.cat((token, b4_local_feat), dim=1))
            local_feat_4 = b4_local_feat[:, 0]

            return global_feat, local_feat_1, local_feat_2, local_feat_3, local_feat_4
        else:
            ###############
            # features = self.backbone(x)
            # backbone_bn_features, backbone_cls_score = self.backbone_classifier(features)
            features = self.backbone(x)

            b1_feat = self.backbone.b1(features)  # [64, 129, 768]
            global_feat = b1_feat[:, 0]  # [64, 768]

            # JPM branch
            feature_length = features.size(1) - 1
            patch_length = feature_length // self.divide_length
            token = features[:, 0:1]
            x = shuffle_unit(features, self.shift_num, self.shuffle_groups)

            # Transformer of Last Layer
            # lf_1
            b1_local_feat = x[:, :patch_length]
            b1_local_feat = self.backbone.b2(torch.cat((token, b1_local_feat), dim=1))
            local_feat_1 = b1_local_feat[:, 0]
            # lf_2
            b2_local_feat = x[:, patch_length : patch_length * 2]
            b2_local_feat = self.backbone.b2(torch.cat((token, b2_local_feat), dim=1))
            local_feat_2 = b2_local_feat[:, 0]
            # lf_3
            b3_local_feat = x[:, patch_length * 2 : patch_length * 3]
            b3_local_feat = self.backbone.b2(torch.cat((token, b3_local_feat), dim=1))
            local_feat_3 = b3_local_feat[:, 0]
            # lf_4
            b4_local_feat = x[:, patch_length * 3 : patch_length * 4]
            b4_local_feat = self.backbone.b2(torch.cat((token, b4_local_feat), dim=1))
            local_feat_4 = b4_local_feat[:, 0]

            backbone_bn_features, _ = self.backbone_classifier(global_feat)
            local_feat_1_bn, _ = self.local_classifier_1(local_feat_1)
            local_feat_2_bn, _ = self.local_classifier_2(local_feat_2)
            local_feat_3_bn, _ = self.local_classifier_3(local_feat_3)
            local_feat_4_bn, _ = self.local_classifier_4(local_feat_4)

            infer_feats = torch.cat([backbone_bn_features, local_feat_1_bn / 4, local_feat_2_bn / 4, local_feat_3_bn / 4, local_feat_4_bn / 4], dim=1)

            return infer_feats
