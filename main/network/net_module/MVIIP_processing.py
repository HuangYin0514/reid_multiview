import torch


class FeatureMapLocation:
    def __init__(self, config):
        super(FeatureMapLocation, self).__init__()
        self.config = config

    def __call__(self, features_map, pids, classifier):
        size = features_map.size(0)
        chunk_size = int(size / 4)
        c, h, w = features_map.size(1), features_map.size(2), features_map.size(3)
        localized_features_map = torch.zeros([size, c, h, w]).to(features_map.device)

        heatmaps = torch.zeros((size, h, w), device=features_map.device)
        for i in range(size):
            classifier_name = []
            classifier_params = []
            for name, param in classifier.named_parameters():
                classifier_name.append(name)
                classifier_params.append(param)
            heatmap_i = torch.matmul(classifier_params[-1][pids[i]].unsqueeze(0), features_map[i].unsqueeze(0).reshape(c, h * w)).detach()
            if heatmap_i.max() != 0:
                heatmap_i = (heatmap_i - heatmap_i.min()) / (heatmap_i.max() - heatmap_i.min())
            heatmap_i = heatmap_i.reshape(h, w)
            heatmap_i = torch.tensor(heatmap_i)
            heatmaps[i, :, :] = heatmap_i

        localized_features_map = features_map * heatmaps.unsqueeze(1).clone().detach()

        return localized_features_map


class FeatureVectorQuantification:
    def __init__(self, config):
        super(FeatureVectorQuantification, self).__init__()
        self.config = config

    def __call__(self, features, cls_scores, pids):
        size = features.size(0)
        prob = torch.log_softmax(cls_scores, dim=1)
        probs = prob[torch.arange(size), pids]
        weights = torch.softmax(probs.view(-1, 2), dim=1).view(-1).clone().detach()
        quantified_features = weights.unsqueeze(1) * features
        return quantified_features

    def test(self, features, cls_scores, pids):
        size = features.size(0)
        quantified_features = 0.5 * features
        return quantified_features


class FeatureMapQuantification:
    def __init__(self, config):
        super(FeatureMapQuantification, self).__init__()
        self.config = config

    def __call__(self, features_map, cls_scores, pids):
        size = features_map.size(0)
        prob = torch.log_softmax(cls_scores, dim=1)
        probs = prob[torch.arange(size), pids]
        weights = torch.softmax(probs.view(-1, 4), dim=1).view(-1).clone().detach()
        quantified_features_map = weights.unsqueeze(1).unsqueeze(2).unsqueeze(3) * features_map
        return quantified_features_map


class FeatureVectorIntegration:
    def __init__(self, config):
        super(FeatureVectorIntegration, self).__init__()
        self.config = config

    def __call__(self, features, pids):
        size = features.size(0)
        chunk_size = int(size / 4)  # 16
        c = features.size(1)

        integrating_features = torch.zeros([chunk_size, c]).to(features.device)
        integrating_pids = torch.zeros([chunk_size]).to(pids.device)

        for i in range(chunk_size):
            integrating_features[i] = 0.5 * (features[4 * i] + features[4 * i + 1] + features[4 * i + 2] + features[4 * i + 3])
            integrating_pids[i] = pids[4 * i]

        return integrating_features, integrating_pids


class FeatureMapIntegration:
    def __init__(self, config):
        super(FeatureMapIntegration, self).__init__()
        self.config = config

    def __call__(self, features_map, pids):
        size = features_map.size(0)
        chunk_size = int(size / 4)  # 16
        c, h, w = features_map.size(1), features_map.size(2), features_map.size(3)

        chunk_features_map = torch.chunk(features_map, chunks=chunk_size, dim=0)
        chunk_pids = torch.chunk(pids, chunks=chunk_size, dim=0)
        integrating_features_map = torch.zeros([chunk_size, c, h, w]).to(features_map.device)
        integrating_pids = torch.zeros([chunk_size]).to(features_map.device)
        for i in range(chunk_size):
            integrating_features_map[i, :, :, :] = chunk_features_map[i][0].unsqueeze(0) + chunk_features_map[i][1].unsqueeze(0) + chunk_features_map[i][2].unsqueeze(0) + chunk_features_map[i][3].unsqueeze(0)
            integrating_pids[i] = chunk_pids[i][0]

        return integrating_features_map, integrating_pids
