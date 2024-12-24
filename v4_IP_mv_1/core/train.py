from network.contrastive_loss import *
from network.processing import *
from tools import CrossEntropyLabelSmooth, KLDivLoss, MultiItemAverageMeter
from tqdm import tqdm


def train(base, loaders, config):
    base.set_train()
    loader = loaders.loader
    meter = MultiItemAverageMeter()
    for epoch, data in enumerate(tqdm(loader)):
        imgs, pids, cids = data
        imgs, pids, cids = imgs.to(base.device), pids.to(base.device).long(), cids.to(base.device).long()
        if config.module == "Lucky":
            ###########################################################
            # Backbone
            features_map = base.model(imgs)
            bn_features = base.model.module.gap_bn(features_map)
            bn_features, cls_score = base.model.module.bn_classifier(bn_features)
            ide_loss = CrossEntropyLabelSmooth().forward(cls_score, pids)

            ###########################################################
            # 定位
            localized_features_map = FeatureMapLocalized(config).__call__(features_map, pids, base.model.module.bn_classifier)
            bn_localized_features = base.model.module.gap_bn2(localized_features_map)
            _, localized_cls_score = base.model.module.bn_classifier2(bn_localized_features)
            # 量化
            quantified_features_map = WeightedFeatureMapLocalized(config).__call__(localized_features_map, localized_cls_score, pids)

            ###########################################################
            # # 聚合
            integrating_features_map, integrating_pids = FeatureMapIntegrating(config).__call__(quantified_features_map, localized_cls_score, pids)
            bn_integrating_features = base.model.module.gap_bn(integrating_features_map)
            bn_integrating_features, integrating_cls_score = base.model.module.bn_classifier2(bn_integrating_features)
            integrating_ide_loss = CrossEntropyLabelSmooth().forward(integrating_cls_score, integrating_pids)

            # 全局对比
            reasoning_loss = ReasoningLoss().forward(bn_features, bn_integrating_features)

            ###########################################################
            # 损失函数
            total_loss = ide_loss + integrating_ide_loss + 0.007 * reasoning_loss

            base.model_optimizer.zero_grad()
            total_loss.backward()
            base.model_optimizer.step()

            meter.update(
                {
                    "pid_loss": ide_loss.data,
                    "decoupling_loss": decoupling_loss.data,
                    "shared_special_loss": shared_special_loss.data,
                    "reasoning_loss": reasoning_loss.data,
                }
            )

    return meter.get_val(), meter.get_str()
