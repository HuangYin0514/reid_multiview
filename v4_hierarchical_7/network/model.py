import torch.nn as nn
import torchvision

from .gem_pool import GeneralizedMeanPoolingP
from .seam import SEAM


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


class TransLayer_Classifier(nn.Module):
    def __init__(self, hindden_dim, pid_num):
        super(TransLayer_Classifier, self).__init__()

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
        return bn_features, cls_score


class TransLayer_classifier_layer(nn.Module):
    def __init__(self, config):
        super(TransLayer_classifier_layer, self).__init__()

        self.num_layer = 3

        input_channel = [256, 256, 256, 256]
        classifier_list = nn.ModuleList()
        for i in range(self.num_layer):
            temp = TransLayer_Classifier(input_channel[i], config.pid_num)
            classifier_list.append(temp)
        self.classifier_list = classifier_list

    def forward(self, xs):
        assert isinstance(xs, (tuple, list))
        assert len(xs) == 3

        score_list = []
        for i in range(self.num_layer):
            _, score = self.classifier_list[i](xs[i])
            score_list.append(score)

        return score_list


class TransLayer_1(nn.Module):
    def __init__(self):
        super().__init__()

        self.num_layer = 3

        kernel_size = [(4, 4), (2, 2), (1, 1), (1, 1)]
        pool_list = nn.ModuleList()
        for i in range(self.num_layer):
            temp = nn.MaxPool2d(kernel_size=kernel_size[i])
            pool_list.append(temp)
        self.pool_list = pool_list

        input_channel = [256, 512, 1024, 2048]
        output_channel = [256, 512, 1024, 2048]
        SEAM_list = nn.ModuleList()
        for i in range(self.num_layer):
            temp = SEAM(c1=input_channel[i], c2=output_channel[i], n=1)
            SEAM_list.append(temp)
        self.SEAM_list = SEAM_list

        input_channel = [256, 512, 1024, 2048]
        output_channel = [256, 256, 256, 256]
        cv1_list = nn.ModuleList()
        for i in range(self.num_layer):
            temp = nn.Sequential(
                nn.Conv2d(input_channel[i], output_channel[i], kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(output_channel[i]),
                nn.ReLU(inplace=True),
            )
            cv1_list.append(temp)
        self.cv1_list = cv1_list

        input_channel = [256, 256, 256, 256]
        output_channel = [256, 256, 256, 256]
        cv1_list2 = nn.ModuleList()
        for i in range(self.num_layer):
            temp = nn.Sequential(
                nn.Conv2d(input_channel[i], output_channel[i], kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(output_channel[i]),
                nn.ReLU(inplace=True),
            )
            cv1_list2.append(temp)
        self.cv1_list2 = cv1_list2

    def forward(self, xs):
        assert isinstance(xs, (tuple, list))
        assert len(xs) == 3

        cv_feats_list = []
        for i in range(self.num_layer):
            cv_feats = self.SEAM_list[i](xs[i])
            cv_feats = self.cv1_list[i](cv_feats)
            cv_feats = self.pool_list[i](cv_feats)
            if i != 0:
                cv_feats = cv_feats + cv_feats_list[i - 1]
            cv_feats = self.cv1_list2[i](cv_feats)
            cv_feats_list.append(cv_feats)
        return cv_feats_list


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
        self.transLayer_classifier = TransLayer_classifier_layer(config)

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
        super(
            AuxiliaryModelClassifier,
            self,
        ).__init__()

    def forward(self, features_map):
        return


class AuxiliaryModel(nn.Module):
    def __init__(self, pid_num):
        super(
            AuxiliaryModel,
            self,
        ).__init__()

    def forward(self, x):
        return x
