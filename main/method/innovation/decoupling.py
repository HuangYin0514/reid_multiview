import torch
import torch.nn as nn
import torch.nn.functional as F

from ..module import weights_init


class Feature_Decoupling_Net(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(Feature_Decoupling_Net, self).__init__()

        # shared branch
        self.mlp1 = nn.Sequential(
            nn.Linear(input_dim, output_dim, bias=False),
            nn.BatchNorm1d(output_dim),
        )
        self.mlp1.apply(weights_init.weights_init_kaiming)

        # special branch
        self.mlp2 = nn.Sequential(
            nn.Linear(input_dim, output_dim, bias=False),
            nn.BatchNorm1d(output_dim),
        )
        self.mlp2.apply(weights_init.weights_init_kaiming)

    def forward(self, features):
        shared_features = self.mlp1(features)
        special_features = self.mlp2(features)
        return shared_features, special_features


class Feature_Fusion_Net(nn.Module):
    def __init__(self, input_dim, output_dim, view_num):
        super(Feature_Fusion_Net, self).__init__()

        self.view_num = view_num

        self.mlp1 = nn.Sequential(
            nn.Linear(input_dim * view_num, output_dim, bias=False),
            nn.BatchNorm1d(output_dim),
        )
        self.mlp1.apply(weights_init.weights_init_kaiming)

    def forward(self, features, pids):
        size = features.size(0)
        c = features.size(1)
        chunk_size = int(size / self.view_num)  # 16

        integrate_features = torch.zeros([chunk_size, c * self.view_num]).to(features.device)
        integrate_pids = torch.zeros([chunk_size]).to(pids.device)

        for i in range(chunk_size):
            integrate_features[i] = torch.cat(
                [
                    features[self.view_num * i].unsqueeze(0),
                    features[self.view_num * i + 1].unsqueeze(0),
                    features[self.view_num * i + 2].unsqueeze(0),
                    features[self.view_num * i + 3].unsqueeze(0),
                ],
                dim=1,
            )
            integrate_pids[i] = pids[self.view_num * i]

        integrate_features = self.mlp1(integrate_features)

        return integrate_features, integrate_pids
