from network.common import *
from network.feature_map_processing import *
from tools import CrossEntropyLabelSmooth, MultiItemAverageMeter
from tqdm import tqdm


def train(base, loaders, config):
    base.set_train()
    loader = loaders.loader
    meter = MultiItemAverageMeter()
    config.lower = 0.02
    config.upper = 0.4
    config.ratio = 0.3
    for epoch, data in enumerate(tqdm(loader)):
        imgs, pids, cids = data
        imgs, pids, cids = imgs.to(base.device), pids.to(base.device).long(), cids.to(base.device).long()
        if config.module == "Lucky":
            features_map = base.model(imgs)
            bn_features = base.model.module.gap_bn(features_map)
            bn_features, cls_score = base.model.module.bn_classifier(bn_features)
            ide_loss = CrossEntropyLabelSmooth().forward(cls_score, pids)

            erasing_features_map = FeatureMapErasing(config).__call__(features_map)
            erasing_bn_features = base.model.module.gap_bn(erasing_features_map)
            erasing_bn_features, cls_score = base.model.module.bn_classifier(erasing_bn_features)
            erasing_loss = CrossEntropyLabelSmooth().forward(cls_score, pids)

            # noising_features_map = FeatureMapNoising(config).__call__(features_map)
            # noising_bn_features = base.model.module.gap_bn(noising_features_map)
            # noising_bn_features, cls_score = base.model.module.bn_classifier(noising_bn_features)
            # noising_loss = CrossEntropyLabelSmooth().forward(cls_score, pids)

            transforming_features_map = FeatureMapTransforming(config).__call__(features_map)
            transforming_bn_features = base.model.module.gap_bn(transforming_features_map)
            transforming_bn_features, cls_score = base.model.module.bn_classifier(transforming_bn_features)
            transforming_loss = CrossEntropyLabelSmooth().forward(cls_score, pids)

            # 总损失
            # total_loss = ide_loss + 0.1 * erasing_loss + 0.15 * noising_loss + 0.1 * transforming_loss
            total_loss = ide_loss + 0.1 * erasing_loss + 0.1 * transforming_loss

            # 反向传播
            base.model_optimizer.zero_grad()
            total_loss.backward()
            base.model_optimizer.step()

            meter.update(
                {
                    "pid_loss": ide_loss.data,
                }
            )

    return meter.get_val(), meter.get_str()
