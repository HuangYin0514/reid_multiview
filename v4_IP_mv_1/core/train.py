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
            bn_features, cls_score = base.model.module.pclassifier(features_map)
            ide_loss = CrossEntropyLabelSmooth().forward(cls_score, pids)

            #################################################################
            # 定位
            localized_features_map = FeatureMapLocation(config).__call__(features_map, pids, base.model.module.classifier)
            localized_bn_features, localized_cls_score = base.model.module.pclassifier2(localized_features_map)
            localized_ide_loss = CrossEntropyLabelSmooth().forward(localized_cls_score, pids)

            #################################################################
            # 蒸馏学习
            localized_reasoning_loss = ReasoningLoss().forward(bn_features, localized_bn_features)

            #################################################################
            # Loss
            total_loss = ide_loss + localized_ide_loss + 0.007 * localized_reasoning_loss

            base.model_optimizer.zero_grad()
            total_loss.backward()
            base.model_optimizer.step()

            meter.update(
                {
                    "pid_loss": ide_loss.data,
                    "localized_ide_loss": localized_ide_loss.data,
                    "localized_reasoning_loss": localized_reasoning_loss.data,
                }
            )

    return meter.get_dict(), meter.get_str()
