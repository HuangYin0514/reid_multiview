import torch
import torch.nn as nn
import torchvision

from .common import weights_init_classifier, weights_init_kaiming
from .gem_pool import GeneralizedMeanPoolingP
from .resnet50 import resnet50


class Classifier(nn.Module):
    def __init__(self, pid_num):
        super(Classifier, self).__init__()
        self.pid_num = pid_num
        self.GAP = GeneralizedMeanPoolingP()
        self.BN = nn.BatchNorm1d(2048)
        self.BN.apply(weights_init_kaiming)

        self.classifier = nn.Linear(2048, self.pid_num, bias=False)
        self.classifier.apply(weights_init_classifier)

    def forward(self, features_map):
        features = self.GAP(features_map)
        bn_features = self.BN(features.squeeze())
        cls_score = self.classifier(bn_features)
        if self.training:
            return bn_features, cls_score
        else:
            return bn_features


class Classifier2(nn.Module):
    def __init__(self, pid_num):
        super(Classifier2, self).__init__()
        self.pid_num = pid_num
        self.GAP = GeneralizedMeanPoolingP()
        self.BN = nn.BatchNorm1d(2048)
        self.BN.apply(weights_init_kaiming)

        self.classifier = nn.Linear(2048, self.pid_num, bias=False)
        self.classifier.apply(weights_init_classifier)

    def forward(self, features_map):
        features = self.GAP(features_map)
        bn_features = self.BN(features.squeeze())
        cls_score = self.classifier(bn_features)
        return bn_features, cls_score


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


class FeatureMapIntegrating:
    def __init__(self, config):
        super(FeatureMapIntegrating, self).__init__()
        self.config = config

    def __call__(self, features_map, pids):
        size = features_map.size(0)
        chunk_size = int(size / 4)
        c, h, w = features_map.size(1), features_map.size(2), features_map.size(3)

        # Fusion
        integrating_features_map = features_map.view(chunk_size, 4, c, h, w)  # (chunk_size, 4, c, h, w)
        integrating_features_map = torch.sum(integrating_features_map, dim=1)
        integrating_pids = pids[::4]
        return integrating_features_map, integrating_pids


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.backbone = Backbone()

        self.classifier = Classifier(config.pid_num)
        self.classifier2 = Classifier2(config.pid_num)
        self.feature_map_integrating = FeatureMapIntegrating(config)

    def forward(self, x, pids=None):
        if self.training:
            x1, x2, x3, x4, features_map = self.backbone(x)
            bn_features, cls_score = self.classifier(features_map)
            integrating_features_map, integrating_pids = self.feature_map_integrating.__call__(features_map, pids)
            integrating_bn_features, integrating_cls_score = self.classifier2(integrating_features_map)
            return cls_score, integrating_cls_score, integrating_cls_score, integrating_pids, bn_features, integrating_bn_features
        else:
            images = x
            flip_images = torch.flip(x, [3]).cuda()
            _, _, _, _, features_map = self.backbone(images)
            _, _, _, _, flip_features_map = self.backbone(flip_images)
            bn_features = self.classifier(features_map)
            flip_bn_features = self.classifier(flip_features_map)
            bn_features = bn_features + flip_bn_features
            return bn_features
