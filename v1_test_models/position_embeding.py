import os
import time

import torch
from torch import nn
from torch.nn import functional as F


class CamEncode(nn.Module):
    def __init__(self, channel=2048, depth=128):
        super(CamEncode, self).__init__()

        self.channel = channel
        self.depth = depth

        self.depthnet = nn.Conv2d(channel, depth + channel, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.depthnet(x)
        depth = x[:, : self.depth].softmax(dim=1)
        x = depth.unsqueeze(1) * x[:, self.depth : (self.depth + self.channel)].unsqueeze(2)
        return x


class Position_embeding(nn.Module):
    def __init__(self):
        super(Position_embeding, self).__init__()
        self.downsample = 16

        self.frustum = self.create_frustum()
        self.D, _, _, _ = self.frustum.shape

    def create_frustum(self):
        # make grid in image plane
        ogfH, ogfW = (256, 128)
        fH, fW = ogfH // self.downsample, ogfW // self.downsample
        ds = torch.arange(*[4.0, 45.0, 1.0], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        D, _, _ = ds.shape
        xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)

        # D x H x W x 3
        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def forward(self, x):
        B, C, H, W = x.shape
        frustum = self.create_frustum()
        return frustum


if __name__ == "__main__":
    model = Position_embeding()
    # model = CamEncode()
    x = torch.rand(2, 2048, 16, 8)
    y = model(x)
    print(y.shape)
