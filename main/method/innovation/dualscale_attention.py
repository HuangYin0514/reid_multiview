import copy
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .. import module
from . import lska, pam_up_samper


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


class Dualscale_Attention(nn.Module):
    def __init__(self, input_dim, out_dim, attention_num=2):
        super(Dualscale_Attention, self).__init__()

        self.attention_num = attention_num

        self.attention_lska = lska.LSKA(input_dim, k_size=7)
        self.attention = BasicConv2d(input_dim, attention_num + 1)
        self.bap = Bap(input_dim, out_dim, attention_num)

    def select_attention(self, attention):
        attention = torch.softmax(attention, dim=1)  # The last one is background.
        return attention[:, : self.attention_num]

    def forward(self, features):
        features = self.attention_lska(features)
        attentions = self.attention(features)
        selected_attentions = self.select_attention(attentions)
        bap_AiF_features, bap_features = self.bap(selected_attentions, features)
        return attentions, selected_attentions, bap_AiF_features, bap_features
