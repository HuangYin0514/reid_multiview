import torch
import torch.nn as nn
from tools import CrossEntropyLabelSmooth, KLDivLoss, ReasoningLoss
from torch.nn import functional as F

from .common import *
from .contrastive_loss import SharedSharedLoss, SharedSpecialLoss, SpecialSpecialLoss
from .gem_pool import GeneralizedMeanPoolingP
from .resnet50 import resnet50


class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        resnet = resnet50(pretrained=True)
        # resnet = torchvision.models.resnet50(pretrained=True)
        # Modifiy backbone
        resnet.layer4[0].downsample[0].stride = (1, 1)
        resnet.layer4[0].conv2.stride = (1, 1)
        # Backbone structure
        self.resnet_conv1 = resnet.conv1
        self.resnet_bn1 = resnet.bn1
        self.resnet_maxpool = resnet.maxpool
        self.resnet_layer1 = resnet.layer1
        self.resnet_layer2 = resnet.layer2
        self.resnet_layer3 = resnet.layer3
        self.resnet_layer4 = resnet.layer4

    def forward(self, x):
        x = self.resnet_conv1(x)
        x = self.resnet_bn1(x)
        x = self.resnet_maxpool(x)

        x1 = x
        x = self.resnet_layer1(x)
        x2 = x
        x = self.resnet_layer2(x)
        x3 = x
        x = self.resnet_layer3(x)
        x4 = x
        x = self.resnet_layer4(x)
        return x1, x2, x3, x4, x


class FeatureMapIntegrating(nn.Module):
    def __init__(self, config):
        super(FeatureMapIntegrating, self).__init__()
        self.config = config

    def __call__(self, bn_features, pids):
        bs, f_dim = bn_features.size(0), bn_features.size(1)
        chunk_bs = int(bs / 4)

        # Fusion
        integrating_bn_features = bn_features.view(chunk_bs, 4, f_dim)  # (chunk_size, 4, c, h, w)
        integrating_bn_features = torch.sum(integrating_bn_features, dim=1)
        integrating_pids = pids[::4]
        return integrating_bn_features, integrating_pids


class FeatureDecoupling(nn.Module):
    def __init__(self, config):
        super(FeatureDecoupling, self).__init__()
        self.config = config

        # shared branch
        ic = 2048
        oc = 1024
        self.mlp1 = nn.Sequential(
            nn.Linear(ic, oc, bias=False),
            nn.BatchNorm1d(oc),
        )
        self.mlp1.apply(weights_init_kaiming)

        # special branch
        self.mlp2 = nn.Sequential(
            nn.Linear(ic, oc, bias=False),
            nn.BatchNorm1d(oc),
        )
        self.mlp2.apply(weights_init_kaiming)

    def forward(self, features):
        shared_features = self.mlp1(features)
        special_features = self.mlp2(features)
        return shared_features, special_features


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.backbone = Backbone()

        # 解耦
        self.decoupling = FeatureDecoupling(config)
        self.decoupling_shared_bn_classifier = BN_Classifier(1024, config.pid_num)
        self.decoupling_special_bn_classifier = BN_Classifier(1024, config.pid_num)

        # 特征融合
        self.feature_integrating = FeatureMapIntegrating(config)

        # 分类
        self.gap_bn = GAP_BN(2048)
        self.bn_classifier = BN_Classifier(2048, config.pid_num)
        self.bn_classifier2 = BN_Classifier(2048, config.pid_num)

    def heatmap(self, x):
        _, _, _, _, features_map = self.backbone(x)
        return features_map

    def make_loss(self, input_features, pids, meter):

        (
            features,
            shared_features,
            special_features,
        ) = input_features

        # IDE
        bn_features, cls_score = self.bn_classifier(features)
        ide_loss = CrossEntropyLabelSmooth().forward(cls_score, pids)

        # 多视角
        integrating_features, integrating_pids = self.feature_integrating(features, pids)
        integrating_bn_features, integrating_cls_score = self.bn_classifier2(integrating_features)
        integrating_ide_loss = CrossEntropyLabelSmooth().forward(integrating_cls_score, integrating_pids)
        integrating_reasoning_loss = ReasoningLoss().forward(bn_features, integrating_bn_features)

        # 特征解耦
        _, shared_cls_score = self.decoupling_shared_bn_classifier(shared_features)
        shared_ide_loss = CrossEntropyLabelSmooth().forward(shared_cls_score, pids)
        _, special_cls_score = self.decoupling_special_bn_classifier(special_features)
        special_ide_loss = CrossEntropyLabelSmooth().forward(special_cls_score, pids)

        num_views = 4
        bs = cls_score.size(0)
        chunk_bs = int(bs / num_views)
        decoupling_loss = 0
        for i in range(chunk_bs):
            shared_feature_i = shared_features[num_views * i : num_views * (i + 1), ...]
            special_feature_i = special_features[num_views * i : num_views * (i + 1), ...]
            # (共享-指定)损失
            sharedSpecialLoss = SharedSpecialLoss().forward(shared_feature_i, special_feature_i)
            # (共享)损失
            sharedSharedLoss = SharedSharedLoss().forward(shared_feature_i)
            # (指定)损失
            # specialSpecialLoss = SpecialSpecialLoss().forward(special_feature_i)
            decoupling_loss += sharedSpecialLoss + 0.01 * sharedSharedLoss

        # 总损失
        total_loss = ide_loss + integrating_ide_loss + 0.007 * integrating_reasoning_loss + decoupling_loss + shared_ide_loss + special_ide_loss

        meter.update(
            {
                "pid_loss": ide_loss.data,
                "integrating_pid_loss": integrating_ide_loss.data,
                "integrating_reasoning_loss": integrating_reasoning_loss.data,
                "decoupling_loss": decoupling_loss.data,
                "shared_ide_loss": shared_ide_loss.data,
                "special_ide_loss": special_ide_loss.data,
            }
        )
        return total_loss

    def forward(self, x, pids=None, meter=None):
        if self.training:
            x1, x2, x3, x4, backbone_features_map = self.backbone(x)
            backbone_features = self.gap_bn(backbone_features_map)
            shared_features, special_features = self.decoupling(backbone_features)
            features = torch.cat([shared_features, special_features], dim=1)

            input_features = [
                features,
                shared_features,
                special_features,
            ]
            total_loss = self.make_loss(input_features=input_features, pids=pids, meter=meter)
            return total_loss
        else:

            def core_func(x):
                x1, x2, x3, x4, backbone_features_map = self.backbone(x)
                backbone_features = self.gap_bn(backbone_features_map)
                shared_features, special_features = self.decoupling(backbone_features)
                features = torch.cat([shared_features, special_features], dim=1)
                bn_features, cls_score = self.bn_classifier(features)
                return bn_features

            bn_features = core_func(x)
            flip_images = torch.flip(x, [3])
            flip_bn_features = core_func(flip_images)
            bn_features = bn_features + flip_bn_features
            return bn_features
