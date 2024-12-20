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
            shared_features, special_features = base.model.module.decoupling(bn_features)
            bn_features = base.model.module.feature_fusion(shared_features, special_features)
            bn_features, cls_score = base.model.module.bn_classifier(bn_features)

            # 定位
            localized_features_map = FeatureMapLocalizedIntegratingNoRelu(config).__call__(features_map, pids, base.model.module.bn_classifier)
            bn_localized_features = base.model.module.gap_bn(localized_features_map)
            _, localized_cls_score = base.model.module.bn_classifier2(bn_localized_features)

            # 量化
            quantified_integrating_features_map, integrating_pids = FeatureMapQuantifiedIntegratingProbLogSoftmaxWeights(config).__call__(localized_features_map, localized_cls_score, pids)
            bn_quantified_integrating_features = base.model.module.gap_bn(quantified_integrating_features_map)
            localized_integrating_bn_features, localized_integrating_cls_score = base.model.module.bn_classifier2(bn_quantified_integrating_features)

            # 特征解耦
            _, shared_cls_score = base.model.module.decoupling_shared_bn_classifier(shared_features)
            shared_ide_loss = CrossEntropyLabelSmooth().forward(shared_cls_score, pids)
            _, special_cls_score = base.model.module.decoupling_special_bn_classifier(special_features)
            special_ide_loss = CrossEntropyLabelSmooth().forward(special_cls_score, pids)

            num_views = 4
            bs = cls_score.size(0)
            chunk_bs = int(bs / num_views)
            decoupling_loss = 0
            for i in range(chunk_bs):
                shared_feature_i = shared_features[num_views * i : num_views * (i + 1), ...]
                special_feature_i = special_features[num_views * i : num_views * (i + 1), ...]
                # (共享-指定)损失
                sharedSpecialLoss = SharedSpecialLoss().forward(shared_feature_i, special_feature_i)
                # (共享)损失
                sharedSharedLoss = SharedSharedLoss().forward(shared_feature_i)
                # (指定)损失
                # specialSpecialLoss = SpecialSpecialLoss().forward(special_feature_i)
                decoupling_loss += sharedSpecialLoss + 0.1 * sharedSharedLoss

            # Loss
            # backbone loss
            ide_loss = CrossEntropyLabelSmooth().forward(cls_score, pids)
            # localized integrating loss
            localized_integrating_ide_loss = CrossEntropyLabelSmooth().forward(localized_integrating_cls_score, integrating_pids)
            # reasoning loss
            localized_integrating_reasoning_loss = ReasoningLoss().forward(bn_features, localized_integrating_bn_features)

            total_loss = ide_loss + localized_integrating_ide_loss + 0.007 * localized_integrating_reasoning_loss + decoupling_loss + shared_ide_loss + special_ide_loss

            base.model_optimizer.zero_grad()
            total_loss.backward()
            base.model_optimizer.step()

            meter.update(
                {
                    "pid_loss": ide_loss.data,
                    "localized_integrating_pid_loss": localized_integrating_ide_loss.data,
                    "localized_integrating_reasoning_loss": localized_integrating_reasoning_loss.data,
                    "decoupling_loss": decoupling_loss.data,
                    "shared_ide_loss": shared_ide_loss.data,
                    "special_ide_loss": special_ide_loss.data,
                }
            )

    return meter.get_val(), meter.get_str()
