import wandb
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
            total_loss = CrossEntropyLabelSmooth().forward(cls_score, pids)

            base.model_optimizer.zero_grad()
            total_loss.backward()
            base.model_optimizer.step()

            meter.update(
                {
                    "pid_loss": total_loss.data,
                    "localized_integrating_ide_loss": total_loss.data,
                    "localized_integrating_reasoning_loss": total_loss.data,
                }
            )

    return meter.get_dict(), meter.get_str()
