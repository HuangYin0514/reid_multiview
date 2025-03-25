import torch
import torch.nn as nn
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

            # D: Decoupling
            localized_features = base.model.module.intergarte_gap(localized_features_map).squeeze()
            _, localized_cls_score = base.model.module.backbone_classifier(localized_features)
            shared_features, specific_features, reconstructed_features = base.model.module.featureDecouplingNet(localized_features)
            decoupling_loss = innovation.decoupling.DecouplingLoss(config).forward(shared_features, specific_features)
            decoupling_loss += nn.MSELoss()(reconstructed_features, localized_features)

            # F: Fusion
            weighted_shared_features = 0.5 * shared_features
            multiview_shared_features, integrating_pids = innovation.multi_view.FeatureIntegration(config).__call__(weighted_shared_features, pids)  ## 共享特征
            integrating_specific_features, integrating_pids = base.model.module.featureIntegrationNet(specific_features, pids)  ## 指定特征

            # F: Fusion
            integrating_features = torch.cat([multiview_shared_features, integrating_specific_features], dim=1)

            # I: IDLoss
            integrating_bn_features, integrating_cls_score = base.model.module.intergarte_classifier(integrating_features)
            integrating_ide_loss = loss_function.CrossEntropyLabelSmooth().forward(integrating_cls_score, integrating_pids)

            #################################################################
            # C: ContrastLoss
            contrast_loss = innovation.multi_view.ContrastLoss(config).__call__(backbone_bn_features, integrating_bn_features)

            #################################################################
            # Total loss
            total_loss = ide_loss + integrating_ide_loss + decoupling_loss + 0.007 * contrast_loss

            base.model_optimizer.zero_grad()
            total_loss.backward()
            base.model_optimizer.step()

            meter.update(
                {
                    "pid_loss": ide_loss.data,
                    "integrating_pid_loss": integrating_ide_loss.data,
                    "decoupling_loss": decoupling_loss.data,
                    "contrast_loss": contrast_loss.data,
                }
            )

    return meter
