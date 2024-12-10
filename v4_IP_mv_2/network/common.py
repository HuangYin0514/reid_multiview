import torch
import torch.nn as nn
import torchvision

from .gem_pool import GeneralizedMeanPoolingP
from .resnet50 import resnet50
from .resnet_ibn_a import resnet50_ibn_a


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_out")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif classname.find("Conv") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find("BatchNorm") != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
    elif classname.find("InstanceNorm") != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        resnet = resnet50(pretrained=True)
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


class FeatureIntegrating(nn.Module):
    def __init__(self, config):
        super(FeatureIntegrating, self).__init__()
        self.config = config

    def __call__(self, bn_features, pids):
        bs, f_dim = bn_features.size(0), bn_features.size(1)
        chunk_bs = int(bs / 4)

        # Fusion
        integrating_bn_features = bn_features.view(chunk_bs, 4, f_dim)  # (chunk_size, 4, f_dim)
        integrating_bn_features = torch.sum(integrating_bn_features, dim=1)
        integrating_pids = pids[::4]
        return integrating_bn_features, integrating_pids


class FeatureFusion(nn.Module):
    def __init__(self, config):
        super(FeatureFusion, self).__init__()
        self.config = config

    def __call__(self, features_1, features_2):
        bs = features_1.size(0)
        out = torch.cat([features_1, features_2], dim=1)
        return out


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
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(oc),
        )
        self.mlp1.apply(weights_init_kaiming)

        # special branch
        self.mlp2 = nn.Sequential(
            nn.Linear(ic, oc, bias=False),
            nn.BatchNorm1d(oc),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(oc),
        )
        self.mlp2.apply(weights_init_kaiming)

    def forward(self, features):
        shared_features = self.mlp1(features)
        special_features = self.mlp2(features)
        return shared_features, special_features


class ReasoningLoss(nn.Module):
    def __init__(self):
        super(ReasoningLoss, self).__init__()

    def forward(self, bn_features, bn_features2):
        new_bn_features2 = torch.zeros(bn_features.size()).cuda()
        for i in range(int(bn_features2.size(0) / 4)):
            new_bn_features2[i * 4 : i * 4 + 4] = bn_features2[i]
        loss = torch.norm((bn_features - new_bn_features2), p=2)
        return loss


class MLPResidualBlock(nn.Module):
    def __init__(self, in_channels, num_layers=1):
        super(MLPResidualBlock, self).__init__()
        net = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(in_channels, in_channels, bias=False),
                nn.BatchNorm1d(in_channels),
            )
            net.append(mlp)
        self.net = net
        self.net.apply(weights_init_kaiming)

    def forward(self, x):
        identity = x
        for mlp in self.net:
            out = mlp(x)
            x = out
        out += identity
        return out


class GAP_BN(nn.Module):
    def __init__(self, channel=2048):
        super(GAP_BN, self).__init__()
        self.GAP = GeneralizedMeanPoolingP()
        # self.GAP = nn.AdaptiveAvgPool2d(1)
        self.BN = nn.BatchNorm1d(channel)
        self.BN.apply(weights_init_kaiming)

    def forward(self, features_map):
        features = self.GAP(features_map)
        bn_features = self.BN(features.squeeze())
        return bn_features


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
