import torch
import torch.nn as nn
import torchvision

from .common import weights_init_classifier, weights_init_kaiming
from .gem_pool import GeneralizedMeanPoolingP
from .resnet50 import resnet50


class AuxiliaryModel(nn.Module):
    def __init__(self, pid_num):
        super(
            AuxiliaryModel,
            self,
        ).__init__()

    def forward(self, x):
        return x


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


def ort(x4):
    # define response loss
    bs, c_x4, h_x4, w_x4 = x4.size()
    loss_diver2 = torch.tensor(0.0, device=x4.device).float()
    response = x4.view(bs, c_x4, -1)  # N*1024*128
    norm = torch.norm(response, dim=2).unsqueeze(2)
    norm_T = torch.einsum("kij->kji", [norm])
    norm_matrix = torch.einsum("kij,kjm->kim", [norm, norm_T])
    # Calculate autocorrelation matrix
    response_T = torch.einsum("kij->kji", [response])
    corr = torch.einsum("kij,kjm->kim", [response, response_T])
    norm_matrix = 1.0 / (norm_matrix + 1e-6)
    # Set diagonal elements to 0
    corr *= 1 - torch.eye(1024, 1024, device=x4.device)
    norm_matrix *= 1 - torch.eye(1024, 1024, device=x4.device)
    # response loss
    loss_diver12 = corr * norm_matrix
    loss_diver22 = torch.pow(loss_diver12, 2)
    loss_diver32 = torch.nansum(loss_diver22, dim=(1, 2))
    loss_diver2 = loss_diver32 / (x4.shape[1] * x4.shape[1])
    return loss_diver2.mean()


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.backbone = Backbone()

    def forward(self, x):
        x1, x2, x3, x4, features_map = self.backbone(x)
        loss_diver2 = ort(x4)

        if self.training:
            return features_map, loss_diver2
        else:
            return features_map
