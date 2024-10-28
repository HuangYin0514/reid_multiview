import os
import time

import torch
from torch.nn import functional as F


class CAM:
    def __init__(self, config):
        super(CAM, self).__init__()
        self.config = config

        self.IMAGENET_MEAN = [0.485, 0.456, 0.406]
        self.IMAGENET_STD = [0.229, 0.224, 0.225]
        self.GRID_SPACING = 10

        self.save_actmap_path = os.path.join(self.output_path, "actmap_dir/")

    def __call__(self, batch_data, model, classifier):
        bs, c, h, w = batch_data.size()
        features_map = model(batch_data)

        outputs = torch.abs(features_map)
        outputs = torch.max(outputs, dim=1, keepdim=True)[0]
        outputs = outputs.squeeze_(1)
        outputs = outputs.view(bs, h * w)
        outputs = F.normalize(outputs, p=2, dim=1)
        outputs = outputs.view(bs, h, w)

        for j in range(bs):
            filename = int(time.time())
            print(filename)
        return
