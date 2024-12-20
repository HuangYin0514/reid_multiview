from network.contrastive_loss import *
from network.processing import (
    FeatureMapLocalizedIntegratingNoRelu,
    FeatureMapQuantifiedIntegratingProbLogSoftmaxWeights,
)
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
            # Baseline
            features_map = base.model(imgs)
            bn_features = base.model.module.gap_bn(features_map)
            bn_features, cls_score = base.model.module.bn_classifier(bn_features)

            # 定位
            localized_features_map = FeatureMapLocalizedIntegratingNoRelu(config).__call__(features_map, pids, base.model.module.bn_classifier)
            bn_localized_features = base.model.module.gap_bn(localized_features_map)
            _, localized_cls_score = base.model.module.bn_classifier2(bn_localized_features)

            # 量化
            quantified_integrating_features_map, integrating_pids = FeatureMapQuantifiedIntegratingProbLogSoftmaxWeights(config).__call__(localized_features_map, localized_cls_score, pids)
            bn_quantified_integrating_features = base.model.module.gap_bn(quantified_integrating_features_map)
            localized_integrating_bn_features, localized_integrating_cls_score = base.model.module.bn_classifier2(bn_quantified_integrating_features)

            # Loss
            ide_loss = CrossEntropyLabelSmooth().forward(cls_score, pids)
            localized_integrating_ide_loss = CrossEntropyLabelSmooth().forward(localized_integrating_cls_score, integrating_pids)
            localized_integrating_reasoning_loss = ReasoningLoss().forward(bn_features, localized_integrating_bn_features)

            total_loss = ide_loss + localized_integrating_ide_loss + 0.007 * localized_integrating_reasoning_loss

            base.model_optimizer.zero_grad()
            total_loss.backward()
            base.model_optimizer.step()

            meter.update(
                {
                    "pid_loss": ide_loss.data,
                    "localized_integrating_pid_loss": localized_integrating_ide_loss.data,
                    "localized_integrating_reasoning_loss": localized_integrating_reasoning_loss.data,
                }
            )

    return meter.get_val(), meter.get_str()
