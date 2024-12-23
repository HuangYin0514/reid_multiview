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
            localized_features_map = FeatureMapLocalizedIntegratingNoRelu(config).__call__(features_map, pids, base.model.module.bn_classifier)
            bn_localized_features = base.model.module.gap_bn2(localized_features_map)
            _, localized_cls_score = base.model.module.bn_classifier2(bn_localized_features)

            # ###########################################################
            # # 解耦
            # shared_features, special_features = base.model.module.decoupling(bn_localized_features)
            # _, shared_cls_score = base.model.module.decoupling_shared_bn_classifier(shared_features)
            # shared_ide_loss = CrossEntropyLabelSmooth().forward(shared_cls_score, pids)
            # _, special_cls_score = base.model.module.decoupling_special_bn_classifier(special_features)
            # special_ide_loss = CrossEntropyLabelSmooth().forward(special_cls_score, pids)
            # bn_localized_features = base.model.module.feature_fusion(shared_features, special_features)
            # # 重构
            # bn_features_reconstruction = base.model.module.decoupling_reconstruction(shared_features, special_features)
            # reconstruction_loss = nn.MSELoss().forward(bn_features_reconstruction, bn_features)

            ###########################################################
            # 聚合
            quantified_features_map = FeatureMapQuantifiedProbLogSoftmaxWeights(config).__call__(localized_features_map, localized_cls_score, pids)
            quantified_bn_features = base.model.module.decoupling_gap_bn(quantified_features_map)

            # 解耦
            shared_features, special_features = base.model.module.decoupling(quantified_bn_features)
            _, shared_cls_score = base.model.module.decoupling_shared_bn_classifier(shared_features)
            shared_ide_loss = CrossEntropyLabelSmooth().forward(shared_cls_score, pids)
            _, special_cls_score = base.model.module.decoupling_special_bn_classifier(special_features)
            special_ide_loss = CrossEntropyLabelSmooth().forward(special_cls_score, pids)

            ###########################################################
            # 损失函数
            # total_loss = ide_loss + localized_integrating_ide_loss + 0.007 * localized_integrating_reasoning_loss + shared_ide_loss + special_ide_loss + reconstruction_loss
            total_loss = ide_loss + shared_ide_loss + special_ide_loss

            base.model_optimizer.zero_grad()
            total_loss.backward()
            base.model_optimizer.step()

            meter.update(
                {
                    "pid_loss": ide_loss.data,
                    "shared_pid_loss": shared_ide_loss.data,
                    "special_pid_loss": special_ide_loss.data,
                }
            )

    return meter.get_val(), meter.get_str()
