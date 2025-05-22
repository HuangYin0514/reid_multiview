# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import module


class Attention_Module(nn.Module):

    def __init__(self, in_cdim_list, out_cdim=2048, internal_cdim=512, attention_num=2):
        super(Attention_Module, self).__init__()

        # Lateral 1x1 convs to reduce channels
        self.lateral_convs = nn.ModuleList()
        for in_cdim in in_cdim_list:
            li = [nn.Conv2d(in_cdim, internal_cdim, 1, 1, 0)]
            self.lateral_convs.append(nn.Sequential(*li))

        # Output 3x3 convs to reduce aliasing
        self.output_convs = nn.ModuleList()
        for in_cdim in in_cdim_list:
            li = [
                nn.Conv2d(internal_cdim, internal_cdim, 3, 1, 1),
            ]
            self.output_convs.append(nn.Sequential(*li))

        self.finnal_conv = nn.Sequential(
            nn.Conv2d(internal_cdim, out_cdim, 1, 1, 0),
            # nn.AdaptiveAvgPool2d(1),
        )

        self.attention_layer = Dualscale_Attention(out_cdim, out_cdim, attention_num)

    def forward(self, inputs):
        # step 1: lateral 1x1 conv
        lateral_features = [lateral_conv(x) for x, lateral_conv in zip(inputs, self.lateral_convs)]

        # step 2: top-down pathway
        pathway_features_list = [None] * (len(lateral_features) - 1)
        reversed_lateral_features = list(reversed(lateral_features))
        for i in range(len(lateral_features) - 1):
            size_in = reversed_lateral_features[i + 1].shape[-2:]
            f_i = reversed_lateral_features[i]
            f_i = F.interpolate(f_i, size=size_in, mode="nearest")
            f_in = reversed_lateral_features[i + 1]
            pathway_features_list[i] = f_in + f_i

        # step 3: final 3x3 conv
        # out_features_list = [output_conv(f) for f, output_conv in zip(pathway_features_list, self.output_convs)]

        # step 4: fusion
        out_features = pathway_features_list[-1]
        out_features = self.finnal_conv(out_features)
        AiF_features, attention_features = self.attention_layer(out_features)
        return AiF_features, attention_features


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
    def __init__(self, in_channels, out_channels, attention_num=2):
        super(Dualscale_Attention, self).__init__()

        self.attention_num = attention_num

        self.attention = BasicConv2d(in_channels, attention_num + 1)
        self.bap = Bap(in_channels, out_channels, attention_num)

    def select_attention(self, attention):
        attention = torch.softmax(attention, dim=1)  # The last one is background.
        return attention[:, : self.attention_num]

    def forward(self, features):
        attentions = self.attention(features)
        selected_attentions = self.select_attention(attentions)
        bap_AiF_features, bap_features = self.bap(selected_attentions, features)
        return bap_AiF_features, bap_features


if __name__ == "__main__":
    B = 64

    feature_maps_1 = torch.randn(B, 256, 64, 32)
    feature_maps_2 = torch.randn(B, 512, 32, 16)
    feature_maps_3 = torch.randn(B, 1024, 16, 8)
    inter_feature_maps = [feature_maps_1, feature_maps_2, feature_maps_3]

    feature_maps_4 = torch.randn(B, 2048, 16, 8)
    attension_feature_maps = inter_feature_maps + [feature_maps_4]

    model = Attention_Module(in_cdim_list=[256, 512, 1024, 2048])
    print(model)

    outs = model(attension_feature_maps)
    print(outs.shape)
