import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gem_pool import GeneralizedMeanPoolingP


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_out")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif classname.find("Conv") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find("BatchNorm") != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
    elif classname.find("InstanceNorm") != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


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
        self.net.apply(weights_init_kaiming)

    def forward(self, x):
        identity = x
        for mlp in self.net:
            out = mlp(x)
            x = out
        out += identity
        return out


class GAP_BN(nn.Module):
    def __init__(self, channel=2048):
        super(GAP_BN, self).__init__()
        self.GAP = GeneralizedMeanPoolingP()
        # self.GAP = nn.AdaptiveAvgPool2d(1)
        self.BN = nn.BatchNorm1d(channel)
        self.BN.apply(weights_init_kaiming)

    def forward(self, features_map):
        features = self.GAP(features_map)
        bn_features = self.BN(features.squeeze())
        return bn_features


class BN_Classifier(nn.Module):
    def __init__(self, channel=2048, pid_num=None):
        super(BN_Classifier, self).__init__()
        self.pid_num = pid_num
        self.BN = nn.BatchNorm1d(channel)
        self.BN.apply(weights_init_kaiming)
        self.classifier = nn.Linear(channel, self.pid_num, bias=False)
        self.classifier.apply(weights_init_classifier)

    def forward(self, features):
        bn_features = self.BN(features.squeeze())
        cls_score = self.classifier(bn_features)
        return bn_features, cls_score


class Classifier(nn.Module):
    def __init__(self, pid_num):
        super(Classifier, self).__init__()
        self.pid_num = pid_num
        self.GAP = GeneralizedMeanPoolingP()
        self.BN = nn.BatchNorm1d(2048)
        self.BN.apply(weights_init_kaiming)

        self.classifier = nn.Linear(2048, self.pid_num, bias=False)
        self.classifier.apply(weights_init_classifier)

    def forward(self, features_map):
        features = self.GAP(features_map)
        bn_features = self.BN(features.squeeze())
        cls_score = self.classifier(bn_features)
        if self.training:
            return bn_features, cls_score
        else:
            return bn_features


class Classifier2(nn.Module):
    def __init__(self, pid_num):
        super(Classifier2, self).__init__()
        self.pid_num = pid_num
        self.GAP = GeneralizedMeanPoolingP()
        self.BN = nn.BatchNorm1d(2048)
        self.BN.apply(weights_init_kaiming)

        self.classifier = nn.Linear(2048, self.pid_num, bias=False)
        self.classifier.apply(weights_init_classifier)

    def forward(self, features_map):
        features = self.GAP(features_map)
        bn_features = self.BN(features.squeeze())
        cls_score = self.classifier(bn_features)
        return bn_features, cls_score


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    Ref: https://github.com/tkipf/pygcn/blob/master/pygcn/models.py
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
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


def cos_distance(embedded_a, embedded_b):
    embedded_a = F.normalize(embedded_a, dim=1)
    embedded_b = F.normalize(embedded_b, dim=1)
    sim = torch.matmul(embedded_a, embedded_b.T)
    return 1 - sim


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout=0.1):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nout)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x

    def _k_reciprocal_neigh(self, initial_rank, i, k1):
        forward_k_neigh_index = initial_rank[i, : k1 + 1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, : k1 + 1]
        fi = torch.where(backward_k_neigh_index == i)[0]
        return forward_k_neigh_index[fi]

    def _compute_rank(self, dist, k=20):
        _, index = dist.cpu().sort(-1, descending=False)
        return index[:, :k].cuda()

    def _constract_graph(self, feat):
        dist = cos_distance(feat, feat)
        initial_rank = self._compute_rank(dist)
        edge = []
        for i in range(feat.size(0)):
            col = self._k_reciprocal_neigh(initial_rank, i, 5).unsqueeze(0)
            row = torch.ones_like(col) * i
            edge.append(torch.cat([col, row], dim=0))
        edge = torch.cat(edge, dim=1)

        num_nodes = feat.size(0)
        adj = torch.zeros(num_nodes, num_nodes).to(feat.device)
        adj[edge[1], edge[0]] = 1
        return adj
