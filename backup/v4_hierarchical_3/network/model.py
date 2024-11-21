import torch.nn as nn
import torchvision

from .common import (
    TransLayer_1,
    TransLayer_classifier,
    weights_init_classifier,
    weights_init_kaiming,
)
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


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.backbone = Backbone()
        self.transLayer_1 = TransLayer_1()
        self.transLayer_classifier = TransLayer_classifier(config)

        self.transLayer_1.apply(weights_init_kaiming)
        self.transLayer_classifier.apply(weights_init_classifier)

    def forward(self, x):
        x1, x2, x3, x4, features_map = self.backbone(x)

        if self.training:
            hierarchical_features_list = self.transLayer_1([x2, x3, x4])
            hierarchical_score_list = self.transLayer_classifier(hierarchical_features_list)
            return features_map, hierarchical_score_list
        else:
            return features_map


class AuxiliaryModelClassifier(nn.Module):
    def __init__(self, pid_num):
        super(AuxiliaryModelClassifier, self).__init__()

    def forward(self, features_map):
        return


class AuxiliaryModel(nn.Module):
    def __init__(self, pid_num):
        super(AuxiliaryModel, self).__init__()

    def forward(self, x):
        return x
