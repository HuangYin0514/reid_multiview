import torch
import torch.nn as nn
import torch.nn.functional as F


class Test_Model(nn.Module):
    def __init__(self):
        super(Test_Model, self).__init__()

        din = 2048
        dout = 2048
        self.mlp = nn.Sequential(
            nn.Linear(din, dout, bias=False),
            nn.BatchNorm1d(dout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out = self.mlp(x)
        return out


if __name__ == "__main__":
    input_1 = torch.randn(64, 2048)
    model = Test_Model()
    output = model(input_1)
    print(output.shape)
