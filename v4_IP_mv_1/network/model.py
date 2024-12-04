import torch
import torch.nn as nn
from tools import CrossEntropyLabelSmooth

from .common import *
from .contrastive_loss import *
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

    def make_loss(self, input_features, pids, cids, epoch, meter):

        (features,) = input_features

        # IDE
        bn_features, cls_score = self.bn_classifier(features)
        ide_loss = CrossEntropyLabelSmooth().forward(cls_score, pids)

        # 多视角
        integrating_features, integrating_pids = self.feature_integrating(features, pids)
        integrating_bn_features, integrating_cls_score = self.bn_classifier2(integrating_features)
        integrating_ide_loss = CrossEntropyLabelSmooth().forward(integrating_cls_score, integrating_pids)
        integrating_reasoning_loss = ReasoningLoss().forward(bn_features, integrating_bn_features)

        # 总损失
        total_loss = ide_loss + integrating_ide_loss + 0.007 * integrating_reasoning_loss

        meter.update(
            {
                "pid_loss": ide_loss.data,
                "integrating_pid_loss": integrating_ide_loss.data,
                "integrating_reasoning_loss": integrating_reasoning_loss.data,
            }
        )
        return total_loss

    def forward(self, x, pids=None, cids=None, epoch=None, meter=None):
        if self.training:
            x1, x2, x3, x4, backbone_features_map = self.backbone(x)
            features = self.gap_bn(backbone_features_map)

            input_features = [
                features,
            ]
            total_loss = self.make_loss(input_features=input_features, pids=pids, cids=cids, epoch=epoch, meter=meter)
            return total_loss
        else:

            def core_func(x):
                x1, x2, x3, x4, backbone_features_map = self.backbone(x)
                features = self.gap_bn(backbone_features_map)
                bn_features, cls_score = self.bn_classifier(features)
                return bn_features

            bn_features = core_func(x)
            flip_images = torch.flip(x, [3])
            flip_bn_features = core_func(flip_images)
            bn_features = bn_features + flip_bn_features
            return bn_features
