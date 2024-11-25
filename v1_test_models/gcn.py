import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    Ref: https://github.com/tkipf/pygcn/blob/master/pygcn/models.py
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + " (" + str(self.in_features) + " -> " + str(self.out_features) + ")"


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nout)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x


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
