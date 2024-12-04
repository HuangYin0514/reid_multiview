import torch
import torch.nn as nn
import torch.nn.functional as F


class Test_Model(nn.Module):
    def __init__(self):
        super(Test_Model, self).__init__()

        din = 2048
        dhidenden = 2048
        dout = 4096
        depth = 0
        feature_layer_list = nn.ModuleList()
        feature_layer_list.append(nn.Linear(din, dhidenden, bias=False))
        for i in range(depth):
            temp = nn.Linear(dhidenden, dhidenden, bias=False)
            feature_layer_list.append(temp)
        feature_layer_list.append(nn.Linear(dhidenden, dout, bias=False))
        self.feature_layer_list = feature_layer_list

    def forward(self, x):
        out = x
        for feature_layer in self.feature_layer_list:
            out = feature_layer(out)
        return out


if __name__ == "__main__":
    input_1 = torch.randn(64, 2048)
    model = Test_Model()
    output = model(input_1)
    print(output.shape)
