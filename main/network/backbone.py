import copy

import torch
import torch.nn as nn

from .net_module import resnet50, resnet50_ibn_a, vit_base_patch16_224_TransReID


class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        model = vit_base_patch16_224_TransReID(img_size=[256, 128], sie_xishu=3.0, local_feature=True, camera=0, view=0, stride_size=[16, 16], drop_path_rate=0.1, drop_rate=0.0, attn_drop_rate=0.0)
        model.load_param()
        self.model = model

        block = self.model.blocks[-1]
        layer_norm = self.model.norm
        self.b1 = nn.Sequential(copy.deepcopy(block), copy.deepcopy(layer_norm))
        self.b2 = nn.Sequential(copy.deepcopy(block), copy.deepcopy(layer_norm))

    def forward(self, x):
        output = self.model(x, None, None)
        return output
