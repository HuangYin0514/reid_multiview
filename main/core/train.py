import torch
from method import innovation, loss_function
from tools import MultiItemAverageMeter
from tqdm import tqdm


def train(base, loaders, config):
    base.set_train()
    loader = loaders.train_loader
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
            ide_loss = loss_function.CrossEntropyLabelSmooth().forward(backbone_cls_score, pids)

            #################################################################
            # P: Positioning
            localized_features_map = innovation.multi_view.FeatureMapLocation(config).__call__(features_map, pids, base.model.module.backbone_classifier)

            # Q: Quantification
            localized_features = base.model.module.intergarte_gap(localized_features_map).squeeze()
            _, localized_cls_score = base.model.module.backbone_classifier(localized_features)
            quantified_localized_features = innovation.multi_view.FeatureQuantification(config).__call__(localized_features, localized_cls_score, pids)

            # F: Fusion
            integrating_features, integrating_pids = innovation.multi_view.FeatureIntegration(config).__call__(quantified_localized_features, pids)

            # I: IDLoss
            integrating_bn_features, integrating_cls_score = base.model.module.intergarte_classifier(integrating_features)
            integrating_ide_loss = loss_function.CrossEntropyLabelSmooth().forward(integrating_cls_score, integrating_pids)

            #################################################################
            # M: Memory
            memory_loss = base.model.module.memoryBank(backbone_bn_features, pids)

            #################################################################
            # Total loss
            total_loss = ide_loss + integrating_ide_loss + 0.3 * memory_loss

            base.model_optimizer.zero_grad()
            total_loss.backward()
            base.model_optimizer.step()

            base.model.module.memoryBank.updateMemory(backbone_bn_features, pids)

            meter.update(
                {
                    "pid_loss": ide_loss.data,
                    "integrating_pid_loss": integrating_ide_loss.data,
                    "memory_loss": memory_loss.data,
                }
            )

    return meter
