import torch


class FeatureMapLocalizedIntegratingNoRelu:
    def __init__(self, config):
        super(FeatureMapLocalizedIntegratingNoRelu, self).__init__()
        self.config = config

    def __call__(self, features_map, pids, classifier):
        size = features_map.size(0)
        c, h, w = features_map.size(1), features_map.size(2), features_map.size(3)
        chunk_size = int(size / 4)

        # Heatmaps
        classifier_params = [param for name, param in classifier.named_parameters()]
        heatmaps = torch.zeros((size, h, w), device="cuda")
        for i in range(size):
            heatmap_i = torch.matmul(classifier_params[-1][pids[i]].unsqueeze(0), features_map[i].unsqueeze(0).reshape(c, h * w)).detach()
            if heatmap_i.max() != 0:
                heatmap_i = (heatmap_i - heatmap_i.min()) / (heatmap_i.max() - heatmap_i.min())
            heatmap_i = heatmap_i.reshape(h, w)
            heatmaps[i] = heatmap_i

        # Localized
        localized_features_map = features_map * heatmaps.unsqueeze(1).clone().detach()

        return localized_features_map
