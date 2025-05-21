# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention_Module(nn.Module):
    def __init__(self, in_cdim_list, out_cdim=512):
        super(Attention_Module, self).__init__()

        # Lateral 1x1 convs to reduce channels
        self.lateral_convs = nn.ModuleList()
        for in_cdim in in_cdim_list:
            li = [nn.Conv2d(in_cdim, out_cdim, 1, 1, 0)]
            self.lateral_convs.append(nn.Sequential(*li))

        # Output 3x3 convs to reduce aliasing
        self.output_convs = nn.ModuleList()
        for in_cdim in in_cdim_list:
            li = [
                nn.Conv2d(out_cdim, out_cdim, 3, 1, 1),
            ]
            self.output_convs.append(nn.Sequential(*li))

        self.finnal_conv = nn.Sequential(
            nn.Conv2d(out_cdim, 2048, 1, 1, 0),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, inputs):
        # step 1: lateral 1x1 conv
        lateral_features = [lateral_conv(x) for x, lateral_conv in zip(inputs, self.lateral_convs)]

        # step 2: top-down pathway
        pathway_features = [None] * (len(lateral_features) - 1)
        reversed_lateral_features = list(reversed(lateral_features))
        for i in range(len(lateral_features) - 1):
            size_in = reversed_lateral_features[i + 1].shape[-2:]
            f_i = reversed_lateral_features[i]
            f_i = F.interpolate(f_i, size=size_in, mode="nearest")
            f_in = reversed_lateral_features[i + 1]
            pathway_features[i] = f_in + f_i

        # step 3: final 3x3 conv
        out_features = [output_conv(f) for f, output_conv in zip(pathway_features, self.output_convs)]

        # step 4: fusion
        out_features = self.finnal_conv(out_features[-1]).squeeze()

        return out_features


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
