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


class SharedSharedLoss(nn.Module):
    def __init__(self):
        super(SharedSharedLoss, self).__init__()

    def forward(self, embedded_a):
        bs = embedded_a.shape[0]

        intergral_embedded_a = torch.mean(embedded_a, dim=0, keepdim=True)
        print(intergral_embedded_a.shape)

        contrastive_loss = 0
        for i in range(bs):
            contrastive_loss += torch.norm((intergral_embedded_a - embedded_a[i]), p=2)

        return torch.mean(contrastive_loss)


class SpecialSpecialLoss(nn.Module):
    def __init__(self):
        super(SpecialSpecialLoss, self).__init__()

    def forward(self, embedded_a):
        bs = embedded_a.shape[0]
        sims = cos_sim(embedded_a, embedded_a)
        mask = ~torch.eye(bs, dtype=torch.bool)  # mask out diagonal
        non_diag_sims = sims[mask]
        loss = -torch.log(1 - non_diag_sims)
        return torch.mean(loss)


class ReasoningLoss(nn.Module):
    def __init__(self):
        super(ReasoningLoss, self).__init__()

    def forward(self, bn_features, bn_features2):
        new_bn_features2 = torch.zeros(bn_features.size()).cuda()
        for i in range(int(bn_features2.size(0) / 4)):
            new_bn_features2[i * 4 : i * 4 + 4] = bn_features2[i]
        loss = torch.norm((bn_features - new_bn_features2), p=2)
        return loss
