from network.contrastive_loss import *
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
            features_map = base.model(imgs)
            features = base.model.module.gap_bn(features_map)

            # IDE
            bn_features, cls_score = base.model.module.bn_classifier(features)
            ide_loss = CrossEntropyLabelSmooth().forward(cls_score, pids)

            # 多视角
            integrating_features, integrating_pids = base.model.module.feature_integrating(features, pids)
            integrating_bn_features, integrating_cls_score = base.model.module.bn_classifier2(integrating_features)
            integrating_ide_loss = CrossEntropyLabelSmooth().forward(integrating_cls_score, integrating_pids)
            integrating_reasoning_loss = ReasoningLoss().forward(bn_features, integrating_bn_features)

            # 总损失
            total_loss = ide_loss + integrating_ide_loss + 0.007 * integrating_reasoning_loss

            base.model_optimizer.zero_grad()
            total_loss.backward()
            base.model_optimizer.step()

            meter.update(
                {
                    "pid_loss": ide_loss.data,
                    "integrating_pid_loss": integrating_ide_loss.data,
                    "integrating_reasoning_loss": integrating_reasoning_loss.data,
                }
            )

    return meter.get_val(), meter.get_str()
