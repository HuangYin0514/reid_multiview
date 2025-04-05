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
    def __init__(self, input_dim, out_dim, num_attention=2):
        super(Dualscale_Attention, self).__init__()

        self.num_attention = num_attention

        self.attention = BasicConv2d(input_dim, num_attention + 1)
        self.bap = Bap(input_dim, out_dim, num_attention)

    def select_attention(self, attention):
        attention = torch.softmax(attention, dim=1)  # The last one is background.
        return attention[:, : self.num_attention]

    def forward(self, features):
        attentions = self.attention(features)
        selected_attentions = self.select_attention(attentions)
        bap_AiF_features, bap_features = self.bap(selected_attentions, features)
        return attentions, selected_attentions, bap_AiF_features, bap_features


class Guide_Dualscale_Attention(Dualscale_Attention):
    def __init__(self, input_dim, out_dim, num_attention=2):
        super(Guide_Dualscale_Attention, self).__init__(input_dim, out_dim, num_attention)
        self.attention_upsample = pam_up_samper.PamUpSamper(num_attention + 1, num_attention + 1, bias=False, scale=1.0)
        self.attention = BasicConv2d(input_dim + num_attention + 1, num_attention + 1)

    def forward(self, features_l3, features_l4, attentions):

        upsample_attentions = self.attention_upsample(attentions)
        features_and_attentions = torch.cat((features_l3, upsample_attentions), dim=1)
        new_attentions = self.attention(features_and_attentions)
        selected_attentions = self.select_attention(new_attentions)

        bap_AiF_features, bap_features = self.bap(selected_attentions, features_l4)
        return selected_attentions, bap_AiF_features, bap_features
