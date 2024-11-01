import torch.nn as nn
import torchvision

from .gem_pool import GeneralizedMeanPoolingP


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_out")
        nn.init.constant_(m.bias, 0.0)
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


class TransLayer_1(nn.Module):
    def __init__(self):
        super().__init__()

        self.c4_net = nn.Sequential(nn.MaxPool2d(kernel_size=(1, 1)), nn.Conv2d(1024, 64, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(64), nn.BatchNorm2d(64))
        self.c3_net = nn.Sequential(nn.MaxPool2d(kernel_size=(2, 2)), nn.Conv2d(512, 64, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(64), nn.BatchNorm2d(64))
        self.c2_net = nn.Sequential(nn.MaxPool2d(kernel_size=(4, 4)), nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(64), nn.BatchNorm2d(64))
        self.c1_net = nn.Sequential(nn.MaxPool2d(kernel_size=(4, 4)), nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(64), nn.BatchNorm2d(64))

        self.c4_net2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(64), nn.BatchNorm2d(64))
        self.c3_net2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(64), nn.BatchNorm2d(64))
        self.c2_net2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(64), nn.BatchNorm2d(64))

    def forward(self, xs):
        assert isinstance(xs, (tuple, list))
        assert len(xs) == 4
        c1, c2, c3, c4 = xs
        c1 = self.c1_net(c1)
        # print(c1.shape)
        # print(c2.shape)
        c2 = self.c2_net2(self.c2_net(c2) + c1)
        c3 = self.c3_net2(self.c3_net(c3) + c2)
        c4 = self.c4_net2(self.c4_net(c4) + c3)
        return (c1, c2, c3, c4)


class TransLayer_Classifier_Base(nn.Module):
    def __init__(self, hindden_dim, pid_num):
        super(
            TransLayer_Classifier_Base,
            self,
        ).__init__()

        self.pid_num = pid_num
        self.GAP = nn.AdaptiveAvgPool2d((1, 1))
        self.BN = nn.BatchNorm1d(hindden_dim)
        self.BN.apply(weights_init_kaiming)

        self.classifier = nn.Linear(hindden_dim, self.pid_num, bias=False)
        self.classifier.apply(weights_init_classifier)

    def forward(self, features_map):
        features = self.GAP(features_map)
        bn_features = self.BN(features.squeeze())
        cls_score = self.classifier(bn_features)
        return cls_score


class TransLayer_classifier(nn.Module):
    def __init__(self, config):
        super(
            TransLayer_classifier,
            self,
        ).__init__()

        self.c4_net_classifier = TransLayer_Classifier_Base(64, config.pid_num)
        self.c3_net_classifier = TransLayer_Classifier_Base(64, config.pid_num)
        self.c2_net_classifier = TransLayer_Classifier_Base(64, config.pid_num)
        self.c1_net_classifier = TransLayer_Classifier_Base(64, config.pid_num)

    def forward(self, xs):
        assert isinstance(xs, (tuple, list))
        assert len(xs) == 4
        c1, c2, c3, c4 = xs
        c1_score = self.c1_net_classifier(c1)
        c2_score = self.c2_net_classifier(c2)
        c3_score = self.c3_net_classifier(c3)
        c4_score = self.c4_net_classifier(c4)
        return (c1_score, c2_score, c3_score, c4_score)


class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        resnet = torchvision.models.resnet50(pretrained=True)
        # Modifiy backbone
        resnet.layer4[0].downsample[0].stride = (1, 1)
        resnet.layer4[0].conv2.stride = (1, 1)
        # Backbone structure
        self.resnet_conv1 = resnet.conv1
        self.resnet_bn1 = resnet.bn1
        self.resnet_relu = resnet.relu
        self.resnet_maxpool = resnet.maxpool
        self.resnet_layer1 = resnet.layer1
        self.resnet_layer2 = resnet.layer2
        self.resnet_layer3 = resnet.layer3
        self.resnet_layer4 = resnet.layer4

    def forward(self, x):
        x = self.resnet_conv1(x)
        x = self.resnet_bn1(x)
        x = self.resnet_relu(x)
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
        super(
            Model,
            self,
        ).__init__()
        self.backbone = Backbone()
        self.transLayer_1 = TransLayer_1()
        self.transLayer_classifier = TransLayer_classifier(config)

    def forward(self, x):
        x1, x2, x3, x4, features_map = self.backbone(x)

        if self.training:
            hierarchical_features_list = self.transLayer_1([x1, x2, x3, x4])
            hierarchical_score_list = self.transLayer_classifier(hierarchical_features_list)
            return features_map, hierarchical_score_list
        else:
            return features_map


class AuxiliaryModel(nn.Module):
    def __init__(self, pid_num):
        super(
            AuxiliaryModel,
            self,
        ).__init__()

    def forward(self, x):
        return x


class AuxiliaryModelClassifier(nn.Module):
    def __init__(self, pid_num):
        super(
            AuxiliaryModelClassifier,
            self,
        ).__init__()

    def forward(self, features_map):
        return features_map


class Classifier(nn.Module):
    def __init__(self, pid_num):
        super(
            Classifier,
            self,
        ).__init__()
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
        super(
            Classifier2,
            self,
        ).__init__()
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
