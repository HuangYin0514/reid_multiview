import torch
import torch.nn as nn

from .net_module import SEAM, resnet50, resnet50_ibn_a


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

        self.seam_layer2 = SEAM(512, 512, 1)
        self.seam_layer3 = SEAM(1024, 1024, 1)

    def forward(self, x):
        x = self.resnet_conv1(x)  # torch.Size([16, 64, 64, 32])
        x = self.resnet_bn1(x)
        x = self.resnet_maxpool(x)

        x1 = x
        x = self.resnet_layer1(x)  # torch.Size([16, 256, 64, 32])
        x2 = x
        x = self.resnet_layer2(x)  # torch.Size([16, 512, 32, 16])
        x3 = x
        x = self.seam_layer2(x)
        x = self.resnet_layer3(x)  # torch.Size([16, 1024, 16, 8])
        x4 = x
        x = self.seam_layer3(x)
        x = self.resnet_layer4(x)  # torch.Size([16, 2048, 16, 8])
        return x1, x2, x3, x4, x
