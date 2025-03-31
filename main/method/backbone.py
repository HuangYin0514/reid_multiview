import copy

import torch
import torch.nn as nn

from .module import resnet50, resnet50_ibn_a


class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        # ----------------- Resnet -----------------
        resnet = resnet50(pretrained=True)
        # resnet = resnet50_ibn_a(pretrained=True)
        # resnet = torchvision.models.resnet50(pretrained=True)

        # ----------------- Modifiy backbone -----------------
        resnet.layer4[0].downsample[0].stride = (1, 1)
        resnet.layer4[0].conv2.stride = (1, 1)

        # ----------------- Backbone structure -----------------
        self.resnet_l1_l2 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
        )
        resnet_l3 = resnet.layer3
        resnet_l4 = resnet.layer4

        # ------------- hard content branch -----------------------
        self.hard_resnet_l3_l4 = nn.Sequential(copy.deepcopy(resnet_l3), copy.deepcopy(resnet_l4))

        # ------------- soft content branch -----------------------
        self.soft_resnet_l3 = nn.Sequential(copy.deepcopy(resnet_l3))
        self.soft_resnet_l4 = nn.Sequential(copy.deepcopy(resnet_l4))

    def forward(self, x):
        features = self.resnet_l1_l2(x)

        # ------------- hard content branch -------------
        hard_features = self.hard_resnet_l3_l4(features)  # [16, 2048, 16, 8])

        # ------------- soft content branch -------------
        soft_features_l3 = self.soft_resnet_l3(features)  # [16, 1024, 16, 8]
        soft_features_l4 = self.soft_resnet_l4(soft_features_l3)  # [16, 2048, 16, 8])

        return hard_features, soft_features_l3, soft_features_l4
