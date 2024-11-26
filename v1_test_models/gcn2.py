import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter


class GNNLayer(Module):
    def __init__(self, in_features_dim, out_features_dim, activation="relu", use_bias=True):
        super(GNNLayer, self).__init__()
        self.in_features = in_features_dim
        self.out_features = out_features_dim
        self.use_bias = use_bias
        self.weight = Parameter(torch.FloatTensor(in_features_dim, out_features_dim))
        if self.use_bias:
            self.bias = Parameter(torch.FloatTensor(out_features_dim))
        self.init_parameters()

        self._bn1d = nn.BatchNorm1d(out_features_dim)
        if activation == "sigmoid":
            self._activation = nn.Sigmoid()
        elif activation == "leakyrelu":
            self._activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == "tanh":
            self._activation = nn.Tanh()
        elif activation == "relu":
            self._activation = nn.ReLU()
        else:
            raise ValueError("Unknown activation type %s" % self._activation)

    def init_parameters(self):
        """Initialize weights"""
        torch.nn.init.xavier_uniform_(self.weight)
        if self.use_bias:
            torch.nn.init.zeros_(self.bias)
        else:
            self.register_parameter("bias", None)

    def forward(self, features, adj, active=True, batchnorm=True):
        support = torch.mm(features, self.weight)
        output = torch.spmm(adj, support)
        if self.use_bias:
            output += self.bias
        if batchnorm:
            output = self._bn1d(output)
        if active:
            output = self._activation(output)
        return output


class GraphEncoder(nn.Module):
    def __init__(self, encoder_dim, activation="relu", batchnorm=True):
        super(GraphEncoder, self).__init__()
        self._dim = len(encoder_dim) - 1
        self._activation = activation
        self._batchnorm = batchnorm

        encoder_layers = []
        for i in range(self._dim):
            encoder_layers.append(GNNLayer(encoder_dim[i], encoder_dim[i + 1], activation=self._activation))
        self._encoder = nn.Sequential(*encoder_layers)

    def forward(self, x, adj, skip_connect=True):

        z = self._encoder[0](x, adj)
        for layer in self._encoder[1:-1]:
            if skip_connect:
                z = layer(z, adj) + z
            else:
                z = layer(z, adj)
        z = self._encoder[-1](z, adj, False, False)
        return z


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, activation=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.activation = activation

    def forward(self, z):
        adj = torch.mm(z, z.t())
        adj = self.activation(adj)
        return adj


if __name__ == "__main__":
    # 测试模型
    num_nodes = 5  # 节点数量
    feature_dim = 2048  # 输入特征维度
    hidden_dim = 2048  # 隐藏层维度
    output_dim = 2048  # 输出维度
    dropout = 0.5  # dropout概率

    # 随机生成输入特征矩阵
    x = torch.randn(num_nodes, feature_dim)

    # 随机生成邻接矩阵 (对称矩阵且有自环)
    adj = torch.randint(0, 2, (num_nodes, num_nodes)).float()
    adj = (adj + adj.T) / 2
    adj[torch.arange(num_nodes), torch.arange(num_nodes)] = 1.0

    # 初始化模型
    model = GCN(nfeat=feature_dim, nhid=hidden_dim, nout=output_dim, dropout=dropout)

    # 前向传播
    output = model(x, adj)
    print("模型输入维度：\n", x.shape)
    print("模型输出维度：", output.shape)
    print(model)
