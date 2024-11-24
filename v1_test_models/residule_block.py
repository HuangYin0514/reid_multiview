import torch
import torch.nn as nn


class MLPResidualBlock(nn.Module):
    def __init__(self, in_channels, num_layers=1):
        super(MLPResidualBlock, self).__init__()

        net = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(in_channels, in_channels, bias=False),
                nn.BatchNorm1d(in_channels),
            )
            net.append(mlp)
        self.net = net

    def forward(self, x):

        identity = x
        for mlp in self.net:
            out = mlp(x)
            x = out
        out += identity

        return out


# 示例：使用残差块
if __name__ == "__main__":
    x = torch.randn(32, 128)
    model = MLPResidualBlock(128, num_layers=1)
    y = model(x)
    print(y.shape)
