import torch
import torch.nn.functional as F
from network import (
    CrossEntropyLabelSmooth,
    DecouplingSharedSharedLoss,
    DecouplingSharedSpecialLoss,
    FeatureMapLocation,
    FeatureRegularizationLoss,
    FeatureVectorIntegration,
    FeatureVectorQuantification,
    TripletLoss,
)
from tools import MultiItemAverageMeter
from tqdm import tqdm


def train(base, loaders, config):
    base.set_train()
    loader = loaders.loader
    meter = MultiItemAverageMeter()
    for epoch, data in enumerate(tqdm(loader)):
        imgs, pids, cids = data
        imgs, pids, cids = imgs.to(base.device), pids.to(base.device).long(), cids.to(base.device).long()
        if config.module == "Lucky":
            #################################################################
            # Baseline
            features = base.model(imgs)
            backbone_bn_features, backbone_cls_score = base.model.module.backbone_classifier(features)
            ide_loss = F.cross_entropy(backbone_cls_score, pids)
            tri_loss = TripletLoss()(features, pids)[0]

            #################################################################
            # Loss
            total_loss = ide_loss + tri_loss

            base.model_optimizer.zero_grad()
            total_loss.backward()
            base.model_optimizer.step()

            meter.update(
                {
                    "pid_loss": ide_loss.data,
                    "tri_loss": tri_loss.data,
                }
            )

    return meter.get_dict(), meter.get_str()
