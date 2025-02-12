import torch
import torch.nn as nn


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1.0 * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(x, y.t(), beta=1, alpha=-2)
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def hard_example_mining(dist_mat, labels, return_inds=False):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    dist_ap, relative_p_inds = torch.max(dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]
    dist_an, relative_n_inds = torch.min(dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
    # shape [N]
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    if return_inds:
        # shape [N, N]
        ind = labels.new().resize_as_(labels).copy_(torch.arange(0, N).long()).unsqueeze(0).expand(N, N)
        # shape [N, 1]
        p_inds = torch.gather(ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
        n_inds = torch.gather(ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
        # shape [N]
        p_inds = p_inds.squeeze(1)
        n_inds = n_inds.squeeze(1)
        return dist_ap, dist_an, p_inds, n_inds

    return dist_ap, dist_an


class TripletLoss(object):
    """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
    Loss for Person Re-Identification'."""

    def __init__(self, margin=None):
        self.margin = margin
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, global_feat, labels, normalize_feature=False):
        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)
        dist_mat = euclidean_dist(global_feat, global_feat)
        dist_ap, dist_an = hard_example_mining(dist_mat, labels)
        y = dist_an.new().resize_as_(dist_an).fill_(1)
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        # return loss, dist_ap, dist_an
        return loss


# class TripletLoss_v2(nn.Module):
#     """Triplet loss with hard positive/negative mining.
#     Reference:
#         Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
#     Imported from `<https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py>`_.
#     Args:
#         margin (float, optional): margin for triplet. Default is 0.3.
#     """

#     def __init__(self, margin=0.3):
#         super(TripletLoss_v2, self).__init__()
#         self.margin = margin
#         self.ranking_loss = nn.MarginRankingLoss(margin=margin)

#     def forward(self, inputs, targets):
#         """
#         Args:
#             inputs (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
#             targets (torch.LongTensor): ground truth labels with shape (num_classes).
#         """
#         n = inputs.size(0)

#         # Compute pairwise distance, replace by the official when merged
#         # ||a-b||^2 = ||a||^2 -2 * <a,b> + ||b||^2
#         dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
#         dist = dist + dist.t()
#         # dist.addmm_(1, -2, inputs, inputs.t())
#         dist = torch.addmm(input=dist, mat1=inputs, mat2=inputs.t(), alpha=-2, beta=1)
#         dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

#         # For each anchor, find the hardest positive and negative
#         mask = targets.expand(n, n).eq(targets.expand(n, n).t())

#         dist_ap, dist_an = [], []

#         for i in range(n):
#             dist_ap.append(dist[i][mask[i] == True].max().unsqueeze(0))
#             dist_an.append(dist[i][mask[i] == False].min().unsqueeze(0))

#         dist_ap = torch.cat(dist_ap)

#         dist_an = torch.cat(dist_an)

#         # Compute ranking hinge loss
#         y = torch.ones_like(dist_an)

#         return self.ranking_loss(dist_an, dist_ap, y)
