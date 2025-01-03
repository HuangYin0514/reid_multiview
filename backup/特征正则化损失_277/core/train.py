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

            #################################################################
            # 解耦
            bn_quantified_features = base.model.module.gap_bn(quantified_features_map)
            shared_features, special_features = base.model.module.decoupling(bn_quantified_features)
            shared_special_features = base.model.module.decoupling_fusion(shared_features, special_features)

            # 3个损失
            # 共享特征损失
            _, shared_cls_score = base.model.module.decoupling_shared_classifier(shared_features)
            shared_ide_loss = CrossEntropyLabelSmooth().forward(shared_cls_score, pids)
            # 指定特征损失
            _, special_cls_score = base.model.module.decoupling_special_classifier(special_features)
            special_ide_loss = CrossEntropyLabelSmooth().forward(special_cls_score, pids)
            # 融合特征损失
            _, fusion_cls_score = base.model.module.decoupling_fusion_classifier(shared_special_features)
            fusion_ide_loss = CrossEntropyLabelSmooth().forward(fusion_cls_score, pids)
            decopling_loss = shared_ide_loss + special_ide_loss + fusion_ide_loss

            localized_integrating_reasoning_loss = FeatureRegularizationLoss().forward(bn_features)

            #################################################################
            # Loss
            total_loss = ide_loss + decopling_loss / (decopling_loss / ide_loss).detach() + localized_integrating_reasoning_loss / (localized_integrating_reasoning_loss / ide_loss).detach()

            base.model_optimizer.zero_grad()
            total_loss.backward()
            base.model_optimizer.step()

            meter.update(
                {
                    "pid_loss": ide_loss.data,
                    "decopling_loss": decopling_loss.data,
                    "localized_integrating_reasoning_loss": localized_integrating_reasoning_loss.data,
                }
            )

    return meter.get_val(), meter.get_str()
