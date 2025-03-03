import torch
from network import (
    CrossEntropyLabelSmooth,
    DecouplingSharedSharedLoss,
    DecouplingSharedSpecialLoss,
    FeatureMapLocation,
    FeatureRegularizationLoss,
    FeatureVectorIntegration,
    FeatureVectorQuantification,
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
            # R: Resnet
            features_map = base.model(imgs)

            #################################################################
            # I: IDLoss
            backbone_features = base.model.module.backbone_gap(features_map).squeeze()
            backbone_bn_features, backbone_cls_score = base.model.module.backbone_classifier(backbone_features)
            ide_loss = CrossEntropyLabelSmooth().forward(backbone_cls_score, pids)

            # R: Regularization
            reasoning_loss = FeatureRegularizationLoss().forward(backbone_bn_features)

            #################################################################
            # F: Fusion
            gap_intergarte_features = base.model.module.intergarte_gap(features_map).squeeze()
            integrating_features, integrating_pids = FeatureVectorIntegration(config).__call__(gap_intergarte_features, pids)

            # I: IDLoss
            integrating_bn_features, integrating_cls_score = base.model.module.intergarte_classifier(integrating_features)
            integrating_ide_loss = CrossEntropyLabelSmooth().forward(integrating_cls_score, integrating_pids)

            #################################################################
            # Total loss
            total_loss = ide_loss + integrating_ide_loss + 0.007 * reasoning_loss

            base.model_optimizer.zero_grad()
            total_loss.backward()
            base.model_optimizer.step()

            meter.update(
                {
                    "pid_loss": ide_loss.data,
                    "reasoning_loss": reasoning_loss.data,
                    "integrating_pid_loss": integrating_ide_loss.data,
                }
            )

    return meter.get_dict(), meter.get_str()
