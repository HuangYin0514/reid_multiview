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
            bn_features, cls_score = base.model.module.classifier(features_map)
            ide_loss = CrossEntropyLabelSmooth().forward(cls_score, pids)

            #################################################################
            # 定位
            localized_features_map = FeatureMapLocation(config).__call__(features_map, pids, base.model.module.classifier)
            _, localized_cls_score = base.model.module.classifier(localized_features_map)

            #################################################################
            # 量化
            quantified_features_map = FeatureMapQuantification(config).__call__(localized_features_map, localized_cls_score, pids)
            integrating_features_map, integrating_pids = FeatureMapIntegration(config).__call__(quantified_features_map, pids)
            localized_integrating_bn_features, localized_integrating_cls_score = base.model.module.classifier2(integrating_features_map)
            localized_integrating_ide_loss = CrossEntropyLabelSmooth().forward(localized_integrating_cls_score, integrating_pids)
            localized_integrating_reasoning_loss = ReasoningLoss().forward(bn_features, localized_integrating_bn_features)

            #################################################################
            # Loss
            total_loss = ide_loss + localized_integrating_ide_loss / (localized_integrating_ide_loss / ide_loss).detach() + localized_integrating_reasoning_loss / (localized_integrating_reasoning_loss / ide_loss).detach()

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
