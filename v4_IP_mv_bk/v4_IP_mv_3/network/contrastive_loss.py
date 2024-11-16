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
        return torch.mean(loss)


class SpecialSpecialLoss(nn.Module):
    def __init__(self):
        super(SpecialSpecialLoss, self).__init__()

    def forward(self, embedded_a):
        sim = cos_sim(embedded_a, embedded_a)
        loss = -torch.log(1 - sim)
        return torch.mean(loss)
