import torch
from method import loss_function, module
from tools import MultiItemAverageMeter
from tqdm import tqdm


def train(base, loaders, config):
    base.set_train()
    loader = loaders.train_loader
    meter = MultiItemAverageMeter()
    for epoch, data in enumerate(tqdm(loader)):
        imgs, pids, cids = data
        imgs, pids, cids = imgs.to(base.device), pids.to(base.device).long(), cids.to(base.device).long()
        if config.module == "Lucky":
            #################################################################
            # R: Resnet
            hard_features, soft_features_l3, soft_features_l4 = base.model(imgs)

            #################################################################
            # H: Hard content branch
            hard_global_embedding = base.model.module.hard_global_embedding(hard_features)
            hard_global_bn_features, hard_global_cls_score = base.model.module.hard_global_head(hard_global_embedding)

            hard_pid_loss = loss_function.CrossEntropyLabelSmooth().forward(hard_global_cls_score, pids)

            #################################################################
            # Total loss
            total_loss = hard_pid_loss

            base.model_optimizer.zero_grad()
            total_loss.backward()
            base.model_optimizer.step()

            meter.update(
                {
                    "hard_pid_loss": hard_pid_loss.data,
                }
            )

    return meter
