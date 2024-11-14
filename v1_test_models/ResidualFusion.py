import copy

import torch
from torch import nn


class _FusionBlock(nn.Module):
    """
    Fusion Block for Residual fusion.
    input -> (_, input_dim) -> norm -> (_, input_dim * expand) -> (_, input_dim) -> output
                   |                                                         |
                   -------------------------------+---------------------------
    """

    expand = 2

    def __init__(self, input_dim, act_func="relu", dropout=0.0, norm_eps=1e-5) -> None:
        super().__init__()
        latent_dim1 = input_dim * self.expand
        latent_dim2 = input_dim // self.expand
        if act_func == "relu":
            self.act = nn.ReLU(inplace=True)
        elif act_func == "tanh":
            self.act = nn.Tanh()
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise ValueError("Activate function must be ReLU or Tanh.")
        self.linear1 = nn.Linear(input_dim, latent_dim1, bias=False)
        self.linear2 = nn.Linear(latent_dim1, input_dim, bias=False)

        self.linear3 = nn.Linear(input_dim, latent_dim2, bias=False)
        self.linear4 = nn.Linear(latent_dim2, input_dim, bias=False)

        self.norm1 = nn.BatchNorm1d(input_dim, eps=norm_eps)
        self.norm2 = nn.BatchNorm1d(input_dim, eps=norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.block1(self.norm1(x))
        x = x + self.block2(self.norm2(x))
        return x

    def block1(self, x):
        return self.linear2(self.dropout1(self.act(self.linear1(x))))

    def block2(self, x):
        return self.linear4(self.dropout2(self.act(self.linear3(x))))


class ResidualFusion(nn.Module):
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
