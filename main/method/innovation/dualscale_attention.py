import copy
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .. import module
from . import pam_up_samper


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BasicConv2d, self).__init__()
        self.attn_num = out_channels

        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.conv.apply(module.weights_init_kaiming)

        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)
        self.conv3.apply(module.weights_init_kaiming)

        self.bn = nn.BatchNorm2d(out_channels)
        self.bn.apply(module.weights_init_kaiming)

    def forward(self, x):
        x1 = self.conv(x)
        x3 = self.conv3(x)
        x = self.bn(x1 + x3)
        return F.relu(x, inplace=True)


class Bap(nn.Module):
    """
    Bap: Bilinear Attention Pooling
    主要功能是对特征图进行加权池化
    input:
        attentions: [batch_size, num_attention, height, width]
        features: [batch_size, in_channels, height, width]
    output:
        AiF_features<List>: num_attention * [batch_size, out_dim]
        bap_features<Tensor>: [batch_size, num_attention * out_dim]
    """

    def __init__(self, input_dim, out_dim, num_attention):
        super(Bap, self).__init__()
        self.pooling = module.GeneralizedMeanPoolingP()

        self.embedding_layer = nn.ModuleList()
        for i in range(num_attention):
            self.embedding_layer.append(module.Embedding(input_dim, out_dim))
        self.embedding_layer.apply(module.weights_init_kaiming)

    def forward(self, attentions, features):
        num_attention = attentions.shape[1]

        AiF_features = []
        for i in range(num_attention):
            AiF = features * attentions[:, i : i + 1, ...]
            AiF = self.embedding_layer[i](AiF)
            pool_AiF = self.pooling(AiF).squeeze()
            AiF_features.append(pool_AiF)

        bap_features = torch.cat(AiF_features, dim=1)
        bap_features = F.relu(bap_features)

        return AiF_features, bap_features


class ASPP(nn.Module):
    """
    Atrous spatial pyramid pooling (ASPP)
    """

    def __init__(self, in_channels=2048):
        super(ASPP, self).__init__()

        atrous_rates = [1, 6, 12]
        kernels = [3, 3, 3]
        hidden_dim = 512

        self.reduce_dim_conv_1 = nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=False)

        self.conv_layers = nn.ModuleList()
        for i in range(len(atrous_rates)):
            rate = atrous_rates[i]
            kernel = kernels[i]
            self.conv_layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel, 1, padding=rate, dilation=rate, bias=True))

        self.conv1 = nn.Conv2d(hidden_dim * 3, in_channels, 1, 1, 0, bias=False)

        self._initialize_weights()

    def forward(self, x):

        out = self.reduce_dim_conv_1(x)

        out_list = []
        for i in range(len(self.conv_layers)):
            out_i = self.conv_layers[i](out)
            out_list.append(out_i)
        out = torch.cat(out_list, dim=1)

        out = self.conv1(out)

        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_in")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                if m.affine:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)


class Dualscale_Attention(nn.Module):
    def __init__(self, in_channels, out_channels, attention_num=2):
        super(Dualscale_Attention, self).__init__()

        self.attention_num = attention_num

        self.aspp = ASPP(in_channels)
        self.attention = BasicConv2d(in_channels, attention_num + 1)
        self.bap = Bap(in_channels, out_channels, attention_num)

    def select_attention(self, attention):
        attention = torch.softmax(attention, dim=1)  # The last one is background.
        return attention[:, : self.attention_num]

    def forward(self, features):
        aspp_features = self.aspp(features)
        attentions = self.attention(aspp_features)
        selected_attentions = self.select_attention(attentions)
        bap_AiF_features, bap_features = self.bap(selected_attentions, features)
        return bap_AiF_features, bap_features
