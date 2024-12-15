import copy

import torch
from torch import nn


class BaseFusion(nn.Module):

    def __init__(self, args, device="cpu") -> None:
        super().__init__()
        self.args = args
        self.device = device


class ResidualFusion(BaseFusion):
    """Fusion block with residual connection.

    A block like to:
     input -> (_, 512) -> (_, 256) -> (_, 512) -> output
       |                                   |
       ------------------+------------------
    """

    def __init__(self, args, device="cpu"):
        super().__init__(args, device)
        act_func = args.fusion.activation
        views = args.views
        num_layers = args.fusion.num_layers
        in_features = args.hidden_dim

        self.use_bn = args.fusion.use_bn

        if self.use_bn:
            self.norm = nn.BatchNorm1d(in_features)

        self.map_layer = nn.Sequential(nn.Linear(in_features * views, in_features, bias=False), nn.BatchNorm1d(in_features), nn.ReLU(inplace=True))
        block = _FusionBlock(in_features, act_func)
        self.fusion_modules = self._get_clones(block, num_layers)

    def forward(self, h):
        h = torch.cat(h, dim=-1)
        # mapping view-specific feature to common feature dim.
        z = self.map_layer(h)
        # fusion.
        for mod in self.fusion_modules:
            z = mod(z)
        if self.use_bn:
            z = self.norm(z)
        return z

    def _get_clones(self, module, N):
        """
        A deep copy will take a copy of the original object and will then recursively take a copy of the inner objects.
        The change in any of the models won’t affect the corresponding model.
        """
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


if __name__ == "__main__":
    model = ResidualFusion(256, 256, kernel_size=3, stride=1, bias=False)
    # model = nn.Conv2d(256, 512, 3, 3, 1)
    x = torch.rand(2, 256, 16, 16)
    y = model(x)
    print(y.shape)
