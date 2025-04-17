import copy

import torch
import torch.nn as nn

from .module import resnet50, resnet50_ibn_a


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
        self.resnet_relu = resnet.relu  # Remove
        self.resnet_maxpool = resnet.maxpool
        self.resnet_layer1 = resnet.layer1
        self.resnet_layer2 = resnet.layer2
        self.resnet_layer3 = resnet.layer3
        self.resnet_layer4 = resnet.layer4

        self.resnet_l0l1l2 = nn.Sequential(
            self.resnet_conv1,
            self.resnet_bn1,
            self.resnet_relu,
            self.resnet_maxpool,
            self.resnet_layer1,
            self.resnet_layer2,
        )

        self.resnet_l3l4 = nn.Sequential(
            self.resnet_layer3,
            self.resnet_layer4,
        )

        self.copy_resnet_l3l4 = nn.Sequential(
            copy.deepcopy(resnet.layer3),
            copy.deepcopy(resnet.layer4),
        )

    def forward(self, x):
        l2_out = self.resnet_l0l1l2(x)
        l4_out = self.resnet_l3l4(l2_out)
        copy_l4_out = self.copy_resnet_l3l4(l2_out)
        return l4_out, copy_l4_out
