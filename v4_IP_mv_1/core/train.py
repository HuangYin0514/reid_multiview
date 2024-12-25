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
            #################################################################
            # Baseline
            features_map = base.model(imgs)
            bn_features, cls_score = base.model.module.classifier(bn_features)
            ide_loss = CrossEntropyLabelSmooth().forward(cls_score, pids)

            #################################################################
            # 定位
            localized_features_map = FeatureMapLocalizedIntegratingNoRelu(config).__call__(features_map, pids, base.model.module.classifier)
            _, localized_cls_score = base.model.module.classifier(localized_features_map)

            #################################################################
            # 量化
            quantified_integrating_features_map, integrating_pids = FeatureMapQuantifiedIntegratingProbLogSoftmaxWeights(config).__call__(localized_features_map, localized_cls_score, pids)
            localized_integrating_bn_features, localized_integrating_cls_score = base.model.module.classifier2(quantified_integrating_features_map)
            localized_integrating_ide_loss = CrossEntropyLabelSmooth().forward(localized_integrating_cls_score, integrating_pids)
            localized_integrating_reasoning_loss = ReasoningLoss().forward(bn_features, localized_integrating_bn_features)

            #################################################################
            # Loss
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
