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

            # F: Fusion
            intergarte_features = base.model.module.intergarte_gap(localized_features_map).squeeze()

            # I: IDLoss
            integrating_bn_features, integrating_cls_score = base.model.module.intergarte_classifier(intergarte_features)
            integrating_ide_loss = loss_function.CrossEntropyLabelSmooth().forward(integrating_cls_score, pids)

            #################################################################
            # C: ContrastLoss
            memory_loss = base.model.module.memoryBank(integrating_bn_features, pids, epoch)

            #################################################################
            # Total loss
            total_loss = ide_loss + integrating_ide_loss + 0.03 * memory_loss

            base.model_optimizer.zero_grad()
            total_loss.backward()
            base.model_optimizer.step()

            meter.update(
                {
                    "pid_loss": ide_loss.data,
                    "integrating_pid_loss": integrating_ide_loss.data,
                    "memory_loss": memory_loss.data,
                }
            )

    return meter
