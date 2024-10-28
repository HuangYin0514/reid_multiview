import argparse
import os

import cv2
import numpy as np
import torch
from torch.nn import functional as F


class CAM:
    def __init__(self, config):
        super(CAM, self).__init__()
        self.config = config

        self.IMAGENET_MEAN = [0.485, 0.456, 0.406]
        self.IMAGENET_STD = [0.229, 0.224, 0.225]
        self.GRID_SPACING = 10

        self.actmap_dir = os.path.join(config.output_path, "actmap/")
        if not os.path.exists(self.actmap_dir):
            os.makedirs(self.actmap_dir)

    def actmap_fn(self, images, model, classifier, pids):
        _, _, height, width = images.shape
        features_map = model(images)
        bs, c, h, w = features_map.shape

        classifier_params = [param for name, param in classifier.named_parameters()]
        heatmaps = torch.zeros((bs, h, w), device="cuda")
        for i in range(bs):
            heatmap_i = torch.matmul(classifier_params[-1][pids[i]].unsqueeze(0), features_map[i].unsqueeze(0).reshape(c, h * w)).detach()
            if heatmap_i.max() != 0:
                heatmap_i = (heatmap_i - heatmap_i.min()) / (heatmap_i.max() - heatmap_i.min())
            heatmap_i = heatmap_i.reshape(h, w)
            heatmaps[i] = heatmap_i

        for j in range(bs):
            filename = str(j)

            # Image
            img = images[j, ...]
            for t, m, s in zip(img, self.IMAGENET_MEAN, self.IMAGENET_STD):
                t.mul_(s).add_(m).clamp_(0, 1)
            img_np = np.uint8(np.floor(img.cpu().detach().numpy() * 255))
            img_np = img_np.transpose((1, 2, 0))  # (c, h, w) -> (h, w, c)

            # Activation map
            am = heatmaps[j, ...].cpu().detach().numpy()
            # am = outputs[j, 2:-2:, 2:-2].numpy()
            am = cv2.resize(am, (width, height))
            am = 255 * (am - np.min(am)) / (np.max(am) - np.min(am) + 1e-12)
            am = np.uint8(np.floor(am))
            am = cv2.applyColorMap(am, cv2.COLORMAP_JET)

            # 重叠图像
            overlapped = img_np * 0.5 + am * 0.5
            overlapped[overlapped > 255] = 255
            overlapped = overlapped.astype(np.uint8)

            # from left to right: original image, activation map, overlapped image
            grid_img = 255 * np.ones((height, 3 * width + 2 * self.GRID_SPACING, 3), dtype=np.uint8)
            grid_img[:, :width, :] = img_np[:, :, ::-1]
            grid_img[:, width + self.GRID_SPACING : 2 * width + self.GRID_SPACING, :] = am
            grid_img[:, 2 * width + 2 * self.GRID_SPACING :, :] = overlapped
            cv2.imwrite(os.path.join(self.actmap_dir, filename + ".jpg"), grid_img)

    def __call__(self, images, model, classifier, pids):
        model.eval()
        classifier.eval()
        self.actmap_fn(images, model, classifier, pids)
        model.train()
        classifier.train()
