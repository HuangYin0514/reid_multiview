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
        self.resnet_maxpool = resnet.maxpool
        self.resnet_layer1 = resnet.layer1
        self.resnet_layer2 = resnet.layer2
        self.resnet_layer3 = resnet.layer3
        self.resnet_layer4 = resnet.layer4

        self.branch2 = nn.Sequential(
            copy.deepcopy(resnet.layer3),
            copy.deepcopy(resnet.layer4),
        )

    def forward(self, x):
        out = self.resnet_conv1(x)
        out = self.resnet_bn1(out)
        out = self.resnet_maxpool(out)

        out1 = out
        out = self.resnet_layer1(out)
        out2 = out
        out = self.resnet_layer2(out)
        out3 = out
        out = self.resnet_layer3(out)
        out4 = out
        out = self.resnet_layer4(out)

        branch2_out = self.branch2(out3)
        return out1, out2, out3, out4, out, branch2_out
