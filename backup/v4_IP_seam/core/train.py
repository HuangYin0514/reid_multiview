from network.processing import (
    FeatureMapIntegrating,
    FeatureMapLocalizedIntegratingNoRelu,
)
from tools import MultiItemAverageMeter
from tqdm import tqdm


def train(base, loaders, config):

    base.set_train()
    loader = loaders.loader
    meter = MultiItemAverageMeter()
    for i, data in enumerate(tqdm(loader)):
        imgs, pids, cids = data
        imgs, pids, cids = imgs.to(base.device), pids.to(base.device).long(), cids.to(base.device).long()
        if config.module == "Lucky":
            cls_score, integrating_cls_score, integrating_cls_score, integrating_pids, bn_features, integrating_bn_features = base.model(imgs, pids)

            ide_loss = base.pid_creiteron(cls_score, pids)
            integrating_ide_loss = base.pid_creiteron(integrating_cls_score, integrating_pids)
            integrating_reasoning_loss = base.reasoning_creiteron(bn_features, integrating_bn_features)

            total_loss = ide_loss + integrating_ide_loss + config.lambda1 * integrating_reasoning_loss

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
