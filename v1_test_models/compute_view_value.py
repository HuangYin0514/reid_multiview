import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import normalize
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter


def compute_view_value(rs, H, view):
    N = H.shape[0]
    w = []
    # all features are normalized
    global_sim = torch.matmul(H, H.t())
    for v in range(view):
        view_sim = torch.matmul(rs[v], rs[v].t())
        related_sim = torch.matmul(rs[v], H.t())
        # The implementation of MMD
        w_v = (torch.sum(view_sim) + torch.sum(global_sim) - 2 * torch.sum(related_sim)) / (N * N)
        w.append(torch.exp(-w_v))
    w = torch.stack(w)
    w = w / torch.sum(w)
    return w.squeeze()


if __name__ == "__main__":
    rs = torch.randn(4, 16, 2048)
    H = torch.randn(16, 2048)
    rs = normalize(rs, dim=1)
    H = normalize(H, dim=1)
    view = 4

    model = compute_view_value

    out = model(rs, H, view)
    print(out)
    print(out.shape)
