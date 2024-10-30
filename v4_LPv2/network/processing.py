import torch


class FeatureMapIntegrating:
    def __init__(self, config):
        super(FeatureMapIntegrating, self).__init__()
        self.config = config

    def __call__(self, features_map, pids):
        size = features_map.size(0)
        chunk_size = int(size / 4)
        c, h, w = features_map.size(1), features_map.size(2), features_map.size(3)

        integrating_features_map = torch.zeros([chunk_size, c, h, w]).cuda()
        integrating_pids = torch.zeros([chunk_size]).cuda()
        chunk_features_map = torch.chunk(features_map, chunk_size)
        chunk_pids = torch.chunk(pids, chunks=chunk_size, dim=0)
        for i in range(chunk_size):
            integrating_features_map[i, :, :, :] = chunk_features_map[i][0].unsqueeze(0) + chunk_features_map[i][1].unsqueeze(0) + chunk_features_map[i][2].unsqueeze(0) + chunk_features_map[i][3].unsqueeze(0)
            integrating_pids[i] = chunk_pids[i][0]
        return integrating_features_map, integrating_pids


class FeatureMapQuantifiedIntegratingProbLogSoftmaxWeights:
    def __init__(self, config):
        super(FeatureMapQuantifiedIntegratingProbLogSoftmaxWeights, self).__init__()
        self.config = config

    def __call__(self, features_map, cls_scores, pids):
        size = features_map.size(0)
        c, h, w = features_map.size(1), features_map.size(2), features_map.size(3)
        chunk_size = int(size / 4)  # 16

        prob = torch.log_softmax(cls_scores, dim=1)
        probs = prob[torch.arange(size), pids]
        weights = torch.softmax(probs.view(-1, 4), dim=1).view(-1).clone().detach()

        quantified_features_map = weights.unsqueeze(1).unsqueeze(2).unsqueeze(3) * features_map

        chunk_quantified_features_map = torch.chunk(quantified_features_map, chunks=chunk_size, dim=0)
        chunk_pids = torch.chunk(pids, chunks=chunk_size, dim=0)
        quantified_integrating_features_map = torch.zeros([chunk_size, c, h, w]).cuda()
        integrating_pids = torch.zeros([chunk_size]).cuda()
        for i in range(chunk_size):
            quantified_integrating_features_map[i, :, :, :] = chunk_quantified_features_map[i][0].unsqueeze(0) + chunk_quantified_features_map[i][1].unsqueeze(0) + chunk_quantified_features_map[i][2].unsqueeze(0) + chunk_quantified_features_map[i][3].unsqueeze(0)
            integrating_pids[i] = chunk_pids[i][0]

        return quantified_features_map, quantified_integrating_features_map, integrating_pids


class FeatureMapLocalizedIntegratingNoRelu:
    def __init__(self, config):
        super(FeatureMapLocalizedIntegratingNoRelu, self).__init__()
        self.config = config

    def __call__(self, features_map, pids, base):
        size = features_map.size(0)
        chunk_size = int(size / 4)
        c, h, w = features_map.size(1), features_map.size(2), features_map.size(3)
        localized_integrating_features_map = torch.zeros([chunk_size, c, h, w]).cuda()
        integrating_pids = torch.zeros([chunk_size]).cuda()
        chunk_pids = torch.chunk(pids, chunks=chunk_size, dim=0)

        # Heatmaps
        classifier_params = [param for name, param in base.classifier.named_parameters()]
        heatmaps = torch.zeros((size, h, w), device="cuda")
        for i in range(size):
            heatmap_i = torch.matmul(classifier_params[-1][pids[i]].unsqueeze(0), features_map[i].unsqueeze(0).reshape(c, h * w)).detach()
            if heatmap_i.max() != 0:
                heatmap_i = (heatmap_i - heatmap_i.min()) / (heatmap_i.max() - heatmap_i.min())
            heatmap_i = heatmap_i.reshape(h, w)
            heatmaps[i] = heatmap_i

        # Localized
        localized_features_map = features_map * heatmaps.unsqueeze(1).clone().detach()

        # Fusion
        chunk_localized_features_map = torch.chunk(localized_features_map, chunk_size)
        for i in range(chunk_size):
            localized_integrating_features_map[i, :, :, :] = chunk_localized_features_map[i][0].unsqueeze(0) + chunk_localized_features_map[i][1].unsqueeze(0) + chunk_localized_features_map[i][2].unsqueeze(0) + chunk_localized_features_map[i][3].unsqueeze(0)
            integrating_pids[i] = chunk_pids[i][0]

        return localized_features_map, localized_integrating_features_map, integrating_pids
