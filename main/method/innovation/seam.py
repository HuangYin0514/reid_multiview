import torch
import torch.nn.functional as F
from torch import nn


class Residual(nn.Module):
    """
    残差模块类，用于实现残差连接。
    """

    def __init__(self, fn):
        super(Residual, self).__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class SEAM(nn.Module):
    """
    SEAM 模块类，用于实现自适应注意力机制。

    参数:
    input_channels (int): 输入通道数。
    output_channels (int): 输出通道数。
    n (int): 残差模块的数量。
    reduction (int): 通道缩减率，默认值为16。
    """

    def __init__(self, input_channels, output_channels, n, reduction=16):
        super(SEAM, self).__init__()
        assert input_channels == output_channels, "input_channels must be equal to output_channels"
        self.DCovN = nn.Sequential(
            # nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1, groups=input_channels),
            # nn.GELU(),
            # nn.BatchNorm2d(output_channels),
            *[
                nn.Sequential(
                    Residual(
                        nn.Sequential(
                            nn.Conv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=3, stride=1, padding=1, groups=output_channels),
                            nn.GELU(),
                            nn.BatchNorm2d(output_channels),
                        )
                    ),
                    nn.Conv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=1, stride=1, padding=0, groups=1),
                    nn.GELU(),
                    nn.BatchNorm2d(output_channels),
                )
                for i in range(n)
            ]
        )
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(output_channels, output_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(output_channels // reduction, output_channels, bias=False),
            nn.Sigmoid(),
        )

        self._initialize_weights()
        # self.initialize_layer(self.avg_pool)
        self.initialize_layer(self.fc)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.DCovN(x)
        y = self.avg_pool(y).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y = torch.exp(y)
        return x * y.expand_as(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=1)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def initialize_layer(self, layer):
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            torch.nn.init.normal_(layer.weight, mean=0.0, std=0.001)
            if layer.bias is not None:
                torch.nn.init.constant_(layer.bias, 0)


if __name__ == "__main__":
    model = SEAM(input_channels=2048, output_channels=2048, n=1)
    x = torch.rand(2, 2048, 16, 8)
    y = model(x)
    print(y.shape)
