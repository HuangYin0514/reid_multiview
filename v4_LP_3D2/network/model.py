import torch
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


class Model(nn.Module):
    def __init__(self):
        super(
            Model,
            self,
        ).__init__()
        resnet = torchvision.models.resnet50(pretrained=True)
        resnet.layer4[0].conv2.stride = (1, 1)
        resnet.layer4[0].downsample[0].stride = (1, 1)

        self.resnet_conv = nn.Sequential(resnet.conv1, resnet.bn1, resnet.maxpool, resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)

    def forward(self, x):
        features_map = self.resnet_conv(x)
        return features_map


class Residual(nn.Module):
    def __init__(self, module, downsample=None):
        super(Residual, self).__init__()
        self.module = module
        self.downsample = downsample
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.module(x)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)


class FCTransform_(nn.Module):
    def __init__(self, image_featmap_size, space_featmap_size):
        super(FCTransform_, self).__init__()
        ic, ih, iw = image_featmap_size  # (256, 16, 16)
        sc, sh, sw = space_featmap_size  # (128, 16, 32)
        self.image_featmap_size = image_featmap_size
        self.space_featmap_size = space_featmap_size
        self.fc_transform = nn.Sequential(nn.Linear(ih * iw, sh * sw), nn.ReLU(), nn.Linear(sh * sw, sh * sw), nn.ReLU())
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=ic, out_channels=sc, kernel_size=1 * 1, stride=1, bias=False),
            nn.BatchNorm2d(sc),
            nn.ReLU(),
        )
        self.residual = Residual(
            module=nn.Sequential(
                nn.Conv2d(in_channels=sc, out_channels=sc, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(sc),
            )
        )

    def forward(self, x):
        x = x.view(
            list(x.size()[:2])
            + [
                self.image_featmap_size[1] * self.image_featmap_size[2],
            ]
        )  # 这个 B,C,H*W
        bev_view = self.fc_transform(x)  # 拿出一个视角
        bev_view = bev_view.view(list(bev_view.size()[:2]) + [self.space_featmap_size[1], self.space_featmap_size[2]])
        bev_view = self.conv1(bev_view)
        bev_view = self.residual(bev_view)
        return bev_view


class AuxiliaryModel(nn.Module):
    def __init__(self, pid_num):
        super(
            AuxiliaryModel,
            self,
        ).__init__()
        self.s32transformer = FCTransform_((2048, 16, 8), (1024, 16, 8))
        self.s64transformer = FCTransform_((1024, 16, 8), (1024, 16, 8))
        self.down = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1024),
        )

    def forward(self, x):
        img2048 = x
        img1024 = self.down(img2048)
        bev_32 = self.s32transformer(img2048)
        bev_64 = self.s64transformer(img1024)
        bev = torch.cat([bev_64, bev_32], dim=1)
        return bev


class AuxiliaryModelClassifier(nn.Module):
    def __init__(self, pid_num):
        super(
            AuxiliaryModelClassifier,
            self,
        ).__init__()

    def forward(self, features_map):
        return


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
