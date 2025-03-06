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

            # R: Regularization
            reasoning_loss = innovation.regularization.FeatureRegularizationLoss().forward(backbone_bn_features)

            #################################################################
            # P: Positioning
            localized_features_map = innovation.multi_view.FeatureMapLocation(config).__call__(features_map, pids, base.model.module.backbone_classifier)

            # Q: Quantification
            localized_features = base.model.module.intergarte_gap(localized_features_map).squeeze()
            _, localized_cls_score = base.model.module.backbone_classifier(localized_features)
            quantified_localized_features = innovation.multi_view.FeatureQuantification(config).__call__(localized_features, localized_cls_score, pids)

            # Decoupling
            shared_features, specific_features = base.model.module.featureDecouplingNet(localized_features)
            decoupling_SharedSpecial_loss = innovation.decoupling.MultiviewSharedSpecialLoss().forward(shared_features, specific_features)
            decoupling_SharedShared_loss = innovation.decoupling.MultiviewSharedSharedLoss().forward(shared_features)

            # F: Fusion
            ## 共享特征
            quantified_shared_features = 0.5 * shared_features
            multiview_shared_features, integrating_pids = innovation.multi_view.MultiviewFeatureIntegration(config).__call__(quantified_shared_features, pids)
            ## 指定特征
            integrating_specific_features, integrating_pids = base.model.module.featureIntegrationNet(specific_features, pids)

            # F: Fusion
            integrating_features = torch.cat([multiview_shared_features, integrating_specific_features], dim=1)

            # I: IDLoss
            integrating_bn_features, integrating_cls_score = base.model.module.intergarte_classifier(integrating_features)
            integrating_ide_loss = loss_function.CrossEntropyLabelSmooth().forward(integrating_cls_score, integrating_pids)

            #################################################################
            # Total loss
            total_loss = ide_loss + integrating_ide_loss + 0.007 * reasoning_loss + decoupling_SharedSpecial_loss + 0.01 * decoupling_SharedShared_loss

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

    return meter
