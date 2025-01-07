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

            # 池化
            localized_features = base.model.module.intergarte_gap(localized_features_map).squeeze()

            # 解耦
            shared_features, specific_features = base.model.module.featureDecoupling(localized_features)
            reconstructed_localized_features = base.model.module.featureReconstruction(shared_features, specific_features)
            decoupling_consistency_loss = DecouplingConsistencyLoss().forward(shared_features, specific_features)

            # 融合
            integrating_feature, integrating_pids = FeatureVectorIntegration(config).__call__(reconstructed_localized_features, pids)

            # 分类
            integrating_bn_features, integrating_cls_score = base.model.module.intergarte_classifier(integrating_feature)
            integrating_ide_loss = CrossEntropyLabelSmooth().forward(integrating_cls_score, integrating_pids)

            #################################################################
            # 蒸馏学习
            reasoning_loss = ReasoningLoss().forward(backbone_bn_features, integrating_bn_features)

            #################################################################
            # Loss
            total_loss = ide_loss + integrating_ide_loss + 0.007 * reasoning_loss + decoupling_consistency_loss

            base.model_optimizer.zero_grad()
            total_loss.backward()
            base.model_optimizer.step()

            meter.update(
                {
                    "pid_loss": ide_loss.data,
                    "localized_ide_loss": integrating_ide_loss.data,
                    "reasoning_loss": reasoning_loss.data,
                }
            )

    return meter.get_dict(), meter.get_str()
