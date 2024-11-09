import torch
from torch.nn import functional as F


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


if __name__ == "__main__":
    model = cosine_distance
    x = torch.rand(2, 256)
    y = torch.rand(2, 256)
    out = model(x, y)
    print(out.shape)
    print(out)
    x = -y
    out2 = torch.cosine_similarity(x, y, dim=1)
    print(out2)
