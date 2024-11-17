import torch
import torch.nn as nn
import torch.nn.functional as F


def cos_sim(embedded_a, embedded_b):
    embedded_a = F.normalize(embedded_a, dim=1)
    embedded_b = F.normalize(embedded_b, dim=1)
    sim = torch.matmul(embedded_a, embedded_b.T)
    return torch.clamp(sim, min=0.0005, max=0.9995)


class SharedSpecialLoss(nn.Module):
    def __init__(self):
        super(SharedSpecialLoss, self).__init__()

    def forward(self, embedded_a, embedded_b):
        sim = cos_sim(embedded_a, embedded_b)
        loss = -torch.log(1 - sim)
        return torch.sum(loss)


class SharedSharedLoss(nn.Module):
    def __init__(self):
        super(SharedSharedLoss, self).__init__()

    def forward(self, embedded_a):
        sims = cos_sim(embedded_a, embedded_a)
        bs = embedded_a.shape[0]
        mask = ~torch.eye(bs, dtype=torch.bool)
        non_diag_sims = sims[mask]
        loss = -torch.log(non_diag_sims)
        return torch.mean(loss)


class SpecialSpecialLoss(nn.Module):
    def __init__(self):
        super(SpecialSpecialLoss, self).__init__()

    def forward(self, embedded_a):
        bs = embedded_a.shape[0]
        sims = cos_sim(embedded_a, embedded_a)
        mask = ~torch.eye(bs, dtype=torch.bool)
        non_diag_sims = sims[mask]
        loss = -torch.log(1 - non_diag_sims)
        return torch.mean(loss)
