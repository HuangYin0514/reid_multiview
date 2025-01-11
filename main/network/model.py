import torch
import torch.nn as nn

from .common import *
from .gem_pool import GeneralizedMeanPoolingP
from .resnet50 import resnet50
from .resnet_ibn_a import resnet50_ibn_a


class FeatureReconstruction(nn.Module):
    def __init__(self, config):
        super(FeatureReconstruction, self).__init__()
        self.config = config

    def __call__(self, features_1, features_2):
        out = torch.cat([features_1, features_2], dim=1)
        return out


class FeatureVectorIntegrationNet(nn.Module):
    def __init__(self, config):
        super(FeatureVectorIntegrationNet, self).__init__()
        self.config = config

        # shared branch
        ic = 1024 * 4
        oc = 1024
        self.mlp1 = nn.Sequential(
            nn.Linear(ic, oc, bias=False),
            nn.BatchNorm1d(oc),
        )
        self.mlp1.apply(weights_init_kaiming)

    def __call__(self, features, pids):
        size = features.size(0)
        c = features.size(1)
        chunk_size = int(size / 4)  # 16

        integrate_features = torch.zeros([chunk_size, c * 4]).to(features.device)
        integrate_pids = torch.zeros([chunk_size]).to(pids.device)

        for i in range(chunk_size):
            integrate_features[i] = torch.cat([features[4 * i].unsqueeze(0), features[4 * i + 1].unsqueeze(0), features[4 * i + 2].unsqueeze(0), features[4 * i + 3].unsqueeze(0)], dim=1)
            integrate_pids[i] = pids[4 * i]

        integrate_features = self.mlp1(integrate_features)

        return integrate_features, integrate_pids


class BN_Classifier(nn.Module):
    def __init__(self, channel=2048, pid_num=None):
        super(BN_Classifier, self).__init__()
        self.pid_num = pid_num
        self.BN = nn.BatchNorm1d(channel)
        self.BN.apply(weights_init_kaiming)
        self.classifier = nn.Linear(channel, self.pid_num, bias=False)
        self.classifier.apply(weights_init_classifier)

    def forward(self, features):
        bn_features = self.BN(features.squeeze())
        cls_score = self.classifier(bn_features)
        return bn_features, cls_score


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


class PClassifier(nn.Module):
    def __init__(self, c_dim, pid_num):
        super(PClassifier, self).__init__()
        self.pid_num = pid_num
        self.GAP = GeneralizedMeanPoolingP()
        self.BN = nn.BatchNorm1d(c_dim)
        self.BN.apply(weights_init_kaiming)

        self.classifier = nn.Linear(c_dim, self.pid_num, bias=False)
        self.classifier.apply(weights_init_classifier)

    def forward(self, features_map):
        features = self.GAP(features_map)
        bn_features = self.BN(features.squeeze())
        cls_score = self.classifier(bn_features)
        return bn_features, cls_score


class Classifier(nn.Module):
    def __init__(self, c_dim, pid_num):
        super(Classifier, self).__init__()
        self.pid_num = pid_num
        self.BN = nn.BatchNorm1d(c_dim)
        self.BN.apply(weights_init_kaiming)

        self.classifier = nn.Linear(c_dim, self.pid_num, bias=False)
        self.classifier.apply(weights_init_classifier)

    def forward(self, features):
        bn_features = self.BN(features.squeeze())
        cls_score = self.classifier(bn_features)
        return bn_features, cls_score


class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        # resnet = resnet50(pretrained=True)
        resnet = resnet50_ibn_a(pretrained=True)
        # resnet = resnet50_ibn_a(pretrained=True)
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

        ####################################
        # IDE
        self.backbone = Backbone()

        ####################################
        # Person classifier [feature map -> bn -> classifier]
        # self.pclassifier = PClassifier(2048, config.pid_num)
        # self.pclassifier2 = PClassifier(2048, config.pid_num)

        ####################################
        # Classifer [bn -> classifier]
        self.backbone_gap = GeneralizedMeanPoolingP()
        self.backbone_classifier = Classifier(2048, config.pid_num)
        self.intergarte_gap = GeneralizedMeanPoolingP()
        self.intergarte_classifier = Classifier(2048, config.pid_num)
        # self.localized_integrating_shared_features_classifier = Classifier(1024, config.pid_num)
        # self.decoupling_shared_classifier = Classifier(1024, config.pid_num)
        # self.decoupling_special_classifier = Classifier(1024, config.pid_num)

        ####################################
        # 解耦
        self.featureDecoupling = FeatureDecoupling(config)
        self.featureReconstruction = FeatureReconstruction(config)

        self.featureVectorIntegrationNet = FeatureVectorIntegrationNet(config)

    def heatmap(self, x):
        _, _, _, _, features_map = self.backbone(x)
        return features_map

    def forward(self, x):
        if self.training:
            x1, x2, x3, x4, features_map = self.backbone(x)
            return features_map
        else:
            ###############
            x1, x2, x3, x4, features_map = self.backbone(x)
            backbone_features = self.backbone_gap(features_map).squeeze()
            backbone_bn_features, backbone_cls_score = self.backbone_classifier(backbone_features)
            return backbone_bn_features
