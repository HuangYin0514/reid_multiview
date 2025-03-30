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
            features_map = base.model(imgs)

            #################################################################
            # I: IDLoss
            backbone_features = base.model.module.backbone_gap(features_map).squeeze()
            backbone_bn_features, backbone_cls_score = base.model.module.backbone_classifier(backbone_features)
            pid_loss = loss_function.CrossEntropyLabelSmooth().forward(backbone_cls_score, pids)

            #################################################################
            # M: Memory
            memory_loss = 0.3 * base.model.module.memoryBank(backbone_bn_features, pids)

            #################################################################
            # Total loss
            total_loss = pid_loss + memory_loss

            base.model_optimizer.zero_grad()
            total_loss.backward()
            base.model_optimizer.step()

            base.model.module.memoryBank.updateMemory(backbone_bn_features, pids)

            meter.update(
                {
                    "pid_loss": pid_loss.data,
                    "memory_loss": memory_loss.data,
                }
            )

    return meter
