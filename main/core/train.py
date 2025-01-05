from network.loss_function import *
from network.processing import *
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
            features_map = base.model(imgs)
            backbone_features = base.model.module.backbone_gap(features_map).squeeze()
            backbone_bn_features, backbone_cls_score = base.model.module.backbone_classifier(backbone_features)
            ide_loss = CrossEntropyLabelSmooth().forward(backbone_cls_score, pids)

            #################################################################
            # 定位
            localized_features_map = FeatureMapLocation(config).__call__(features_map, pids, base.model.module.backbone_classifier)

            # 解耦
            localized_global_features = base.model.module.backbone_gap(localized_features_map).squeeze()
            localized_shared_features, localized_specific_features = base.model.module.featureDecoupling(localized_global_features)

            # 集成
            localized_integrating_shared_features, localized_integrating_shared_pids = FeatureVectorIntegration(config).__call__(localized_shared_features, pids)
            _, localized_integrating_shared_features_scores = base.model.module.localized_integrating_shared_features_classifier(localized_integrating_shared_features)
            localized_integrating_shared_features_loss = CrossEntropyLabelSmooth().forward(localized_integrating_shared_features_scores, localized_integrating_shared_pids)

            # 重建
            localized_reconstructed_features = base.model.module.featureReconstruction(localized_shared_features, localized_specific_features)
            localized_reconstructed_bn_features, localized_reconstructed_scores = base.model.module.intergarte_reconstructed_classifier(localized_reconstructed_features)
            localized_ide_loss = CrossEntropyLabelSmooth().forward(localized_reconstructed_scores, pids)

            #################################################################
            # 蒸馏学习
            reasoning_loss = ReasoningLoss().forward(backbone_bn_features, localized_reconstructed_bn_features)

            #################################################################
            # Loss
            total_loss = ide_loss + localized_ide_loss + 0.007 * reasoning_loss + localized_integrating_shared_features_loss

            base.model_optimizer.zero_grad()
            total_loss.backward()
            base.model_optimizer.step()

            meter.update(
                {
                    "pid_loss": ide_loss.data,
                    "localized_ide_loss": localized_ide_loss.data,
                    "reasoning_loss": reasoning_loss.data,
                    "localized_integrating_shared_features_loss": localized_integrating_shared_features_loss.data,
                }
            )

    return meter.get_dict(), meter.get_str()
