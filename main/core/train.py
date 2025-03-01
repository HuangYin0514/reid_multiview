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
            # Resnet
            features_map = base.model(imgs)

            #################################################################
            # IDLoss
            backbone_features = base.model.module.backbone_pooling(features_map).squeeze()
            backbone_bn_features, backbone_cls_score = base.model.module.backbone_classifier(backbone_features)
            ide_loss = CrossEntropyLabelSmooth().forward(backbone_cls_score, pids)

            #################################################################
            # Positioning
            localized_features_map = FeatureMapLocation(config).__call__(features_map, pids, base.model.module.backbone_classifier)

            # Attention
            localized_features_map = base.model.module.seam_attention(localized_features_map)

            localized_features = base.model.module.intergarte_pooling(localized_features_map).squeeze()  # Pooling 池化
            base.model.module.backbone_classifier.BN(localized_features)  # BN, 影响分类器中统计量，不影响特征
            # _, localized_cls_score = base.model.module.backbone_classifier(localized_features)

            # Decoupling
            shared_features, specific_features = base.model.module.featureDecoupling(localized_features)
            decoupling_SharedSpecial_loss = DecouplingSharedSpecialLoss().forward(shared_features, specific_features)
            decoupling_SharedShared_loss = DecouplingSharedSharedLoss().forward(shared_features)

            # Fusion
            ## 共享特征
            quantified_shared_features = 0.5 * shared_features
            integrating_shared_features, integrating_pids = FeatureVectorIntegration(config).__call__(quantified_shared_features, pids)
            ## 指定特征
            # quantified_specific_features = FeatureVectorQuantification(config).__call__(specific_features, localized_cls_score, pids)
            integrating_specific_features, integrating_pids = base.model.module.featureVectorIntegrationNet(specific_features, pids)
            integrating_features = torch.cat([integrating_shared_features, integrating_specific_features], dim=1)

            # IDLoss
            integrating_bn_features, integrating_cls_score = base.model.module.intergarte_classifier(integrating_features)
            integrating_ide_loss = CrossEntropyLabelSmooth().forward(integrating_cls_score, integrating_pids)

            #################################################################
            # Regularization
            # reasoning_loss = ReasoningLoss().forward(backbone_bn_features, integrating_bn_features)
            reasoning_loss = FeatureRegularizationLoss().forward(backbone_bn_features)

            #################################################################
            # Loss
            total_loss = ide_loss + integrating_ide_loss + 0.007 * reasoning_loss + decoupling_SharedSpecial_loss + 0.01 * decoupling_SharedShared_loss

            base.model_optimizer.zero_grad()
            total_loss.backward()
            base.model_optimizer.step()

            meter.update(
                {
                    "pid_loss": ide_loss.data,
                    "localized_ide_loss": integrating_ide_loss.data,
                    "reasoning_loss": reasoning_loss.data,
                    "decoupling_SharedSpecial_loss": decoupling_SharedSpecial_loss.data,
                    "decoupling_SharedShared_loss": decoupling_SharedShared_loss.data,
                }
            )

    return meter.get_dict(), meter.get_str()
