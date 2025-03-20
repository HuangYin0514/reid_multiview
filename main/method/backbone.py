import torch
import torch.nn as nn

from . import innovation
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

        # self.attention_layer2 = innovation.srm.SRM(512)
        # self.attention_layer3 = innovation.srm.SRM(1024)
        # self.attention_layer2 = innovation.triplet_attention.TripletAttention(kernel_size=5)
        # self.attention_layer3 = innovation.triplet_attention.TripletAttention(kernel_size=5)
        # self.attention_layer2 = innovation.lct.LCT(512, 8)
        # self.attention_layer3 = innovation.lct.LCT(1024, 8)
        # self.attention_layer2 = innovation.gct.GCT(512)
        # self.attention_layer3 = innovation.gct.GCT(1024)
        # self.attention_layer2 = innovation.gc_module.GCModule(512)
        # self.attention_layer3 = innovation.gc_module.GCModule(1024)
        # self.attention_layer2 = innovation.eca.ECALayer(512)
        # self.attention_layer3 = innovation.eca.ECALayer(1024)
        self.attention_layer2 = innovation.dual_attention.CAM()
        self.attention_layer3 = innovation.dual_attention.CAM()
        # self.attention_layer2 = innovation.dual_attention.PAM()
        # self.attention_layer3 = innovation.dual_attention.PAM()
        # self.attention_layer2 = innovation.dual_attention.CAM()
        # self.attention_layer3 = innovation.dual_attention.PAM()
        # self.attention_layer2 = innovation.dual_attention.PAM()
        # self.attention_layer3 = innovation.dual_attention.CAM()

    def forward(self, x):
        x = self.resnet_conv1(x)
        x = self.resnet_bn1(x)
        x = self.resnet_maxpool(x)

        x1 = x
        x = self.resnet_layer1(x)
        x2 = x
        x = self.resnet_layer2(x)
        x3 = x
        x = self.attention_layer2(x)
        x = self.resnet_layer3(x)
        x4 = x
        x = self.attention_layer3(x)
        x = self.resnet_layer4(x)
        return x1, x2, x3, x4, x
