from network.common import *
from network.feature_map_processing import *
from tools import CrossEntropyLabelSmooth, MultiItemAverageMeter
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
            bn_features = base.model.module.gap_bn(features_map)

            # IDE
            bn_features, cls_score = base.model.module.bn_classifier(bn_features)
            ide_loss = CrossEntropyLabelSmooth().forward(cls_score, pids)

            # 总损失
            total_loss = ide_loss

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
