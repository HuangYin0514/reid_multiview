import torch

# class FeatureFusion:
#     def __init__(self, config):
#         super(FeatureFusion, self).__init__()
#         self.config = config

#     def __call__(self, features, pids):
#         size = features.size(0)
#         chunk_size = int(size / 4)  # 16
#         c = features.size(1)

#         integrating_features = torch.zeros([chunk_size, c]).to(features.device)
#         integrating_pids = torch.zeros([chunk_size], dtype=torch.long).to(pids.device)

#         for i in range(chunk_size):
#             integrating_features[i] = 1 * (features[4 * i] + features[4 * i + 1] + features[4 * i + 2] + features[4 * i + 3])
#             integrating_pids[i] = pids[4 * i]

#         return integrating_features, integrating_pids


# class FeatureFusion:
#     def __init__(self, config):
#         super(FeatureFusion, self).__init__()
#         self.config = config

#     def __call__(self, features: torch.Tensor, pids: torch.Tensor, weights: torch.Tensor):
#         """
#         聚合特征，根据 pids 进行加权平均。

#         :param features: (N, D) 维的特征张量，N 是样本数，D 是特征维度
#         :param pids: (N,) 维的张量，表示每个样本的行人 ID
#         :param weights: (N, 1) 维的张量，表示每个样本的权重
#         :return: (M, D) 维的聚合特征张量和 (M,) 维的聚合 pids
#         """
#         unique_pids = pids.unique()  # 更改了pids的顺序
#         fusion_features_list = []

#         for pid in unique_pids:
#             mask = pids == pid
#             selected_features = features[mask]
#             selected_weights = weights[mask]

#             # 加权平均计算聚合特征
#             fusion_feature = (selected_features * selected_weights).sum(dim=0) / selected_weights.sum()
#             fusion_features_list.append(fusion_feature)

#         fusion_features = torch.stack(fusion_features_list)  # (M, D)
#         return fusion_features, unique_pids


class FeatureFusion:
    def __init__(self, config):
        super(FeatureFusion, self).__init__()
        self.config = config

    def __call__(self, features: torch.Tensor, pids: torch.Tensor, weights: torch.Tensor):
        """
        聚合特征，根据 pids 进行加权平均。

        :param features: (N, D) 维的特征张量，N 是样本数，D 是特征维度
        :param pids: (N,) 维的张量，表示每个样本的行人 ID
        :param weights: (N, 1) 维的张量，表示每个样本的权重
        :return: (M, D) 维的聚合特征张量和 (M,) 维的聚合 pids
        """
        unique_pids = pids.unique()  # 更改了pids的顺序
        fusion_features_list = []

        for pid in unique_pids:
            mask = pids == pid
            selected_features = features[mask]
            selected_weights = weights[mask]

            # 加权平均计算聚合特征
            fusion_feature = (selected_features * selected_weights).sum(dim=0) / selected_weights.sum()
            fusion_features_list.append(fusion_feature.clone().detach())

        fusion_features = torch.stack(fusion_features_list)  # (M, D)
        return fusion_features, unique_pids
