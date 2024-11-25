import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_distance(input1, input2):
    """Computes cosine distance.

    Args:
        input1 (torch.Tensor): 2-D feature matrix.
        input2 (torch.Tensor): 2-D feature matrix.

    Returns:
        torch.Tensor: distance matrix.
    """
    input1_normed = F.normalize(input1, p=2, dim=1)
    input2_normed = F.normalize(input2, p=2, dim=1)
    distmat = 1 - torch.mm(input1_normed, input2_normed.t())
    return distmat


class Cgnet(nn.Module):
    def __init__(self, num_classes, out_channels, k=5, normalize=True, bias=True, alpha=0.3, pretrained=True):
        super().__init__()
        self.k = k

        self.gnn1 = GINConv(self.mlp, eps=alpha, train_eps=False)
        self.classifier_layer = nn.Linear(2048, num_classes, bias=False)
        self.out_channels = out_channels
        self.bottleneck = nn.BatchNorm1d(2048)
        self.bottleneck.bias.requires_grad_(False)  # no shift

        self.reset_parameters()
        self.classifier_layer.apply(weights_init_classifier)

    def _construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
        """Constructs fully connected layer

        Args:
            fc_dims (list or tuple): dimensions of fc layers, if None, no fc layers are constructed
            input_dim (int): input dimension
            dropout_p (float): dropout probability, if None, dropout is unused
        """
        if fc_dims is None:
            self.feature_dim = input_dim
            return None

        assert isinstance(fc_dims, (list, tuple)), "fc_dims must be either list or tuple, but got {}".format(type(fc_dims))

        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout_p is not None:
                layers.append(nn.Dropout(p=dropout_p))
            input_dim = dim

        self.feature_dim = fc_dims[-1]

        return nn.Sequential(*layers)

    def reset_parameters(self):

        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def _k_reciprocal_neigh(self, initial_rank, i, k1):
        forward_k_neigh_index = initial_rank[i, : k1 + 1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, : k1 + 1]
        fi = torch.where(backward_k_neigh_index == i)[0]
        return forward_k_neigh_index[fi]

    def _compute_rank(self, dist, k=20):
        _, index = dist.cpu().sort(-1, descending=False)
        return index[:, :k].cuda()

    def _constract_graph(self, feat):
        dist = cosine_distance(feat, feat)
        initial_rank = self._compute_rank(dist)
        edge = []
        for i in range(feat.size(0)):
            col = self._k_reciprocal_neigh(initial_rank, i, self.k).unsqueeze(0)
            row = torch.ones_like(col) * i
            edge.append(torch.cat([col, row], dim=0))
        edge = torch.cat(edge, dim=1)
        return edge

    def forward(self, x, return_featuremaps=False):
        classifier, cnn_feature, f = self.cnn(x)
        if return_featuremaps:
            return f
        if not self.training:
            return cnn_feature
        edge_index_c = self._constract_graph(cnn_feature)
        gnn_feature = self.gnn1(cnn_feature, edge_index_c)
        clsfeat = self.bottleneck(gnn_feature)
        gnn_feature_c = self.classifier_layer(clsfeat)
        return gnn_feature_c, classifier, gnn_feature, cnn_feature

    def getGarph(self, cnn_feature, edge_index=None):
        gnn_feature = self.gnn1(cnn_feature, edge_index)
        gnn_feature = self.bottleneck(gnn_feature)
        return gnn_feature


if __name__ == "__main__":
    print("main")
