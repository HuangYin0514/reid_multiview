import torch
from method import innovation, loss_function, module
from tools import MultiItemAverageMeter
from tqdm import tqdm


def train(base, loaders, config):
    base.set_train()
    meter = MultiItemAverageMeter()
    for epoch, data in enumerate(tqdm(loaders.train_loader)):
        imgs, pids, cids = data
        imgs, pids, cids = imgs.to(base.device), pids.to(base.device).long(), cids.to(base.device).long()
        if config.MODEL.MODULE == "Lucky":
            total_loss = 0.0

            # R: Resnet
            resnet_feature_maps = base.model(imgs)

            # ------------- Global content branch -----------------------
            global_features = base.model.module.global_pooling(resnet_feature_maps).squeeze()
            global_bn_features, global_cls_score = base.model.module.global_classifier(global_features)
            global_pid_loss = loss_function.CrossEntropyLabelSmooth().forward(global_cls_score, pids)
            total_loss += global_pid_loss

            # ------------- Multiview content branch  -----------------------
            # Positioning
            multiview_localized_features_map = base.model.module.multiview_feature_map_location(resnet_feature_maps, pids, base.model.module.global_classifier)

            # Quantification
            multiview_localized_features = base.model.module.multiview_pooling(multiview_localized_features_map).squeeze()
            _, multiview_localized_cls_score = base.model.module.global_classifier(multiview_localized_features)
            multiview_quantified_localized_features = base.model.module.multiview_feature_quantification(
                multiview_localized_features, multiview_localized_cls_score, pids
            )

            # Fusion
            multiview_fusion_features, multiview_fusion_pids = base.model.module.multiview_feature_fusion(multiview_quantified_localized_features, pids)

            multiview_fusion_bn_features, multiview_cls_score = base.model.module.multiview_classifier(multiview_fusion_features)
            multiview_pid_loss = loss_function.CrossEntropyLabelSmooth().forward(multiview_cls_score, multiview_fusion_pids)
            total_loss += multiview_pid_loss

            # ------------- ContrastLoss  -----------------------
            # contrast_loss = base.model.module.contrast_loss(global_bn_features, multiview_fusion_bn_features)
            contrast_loss = base.model.module.contrast_kl_loss(global_cls_score, multiview_cls_score)
            total_loss += contrast_loss

            base.model_optimizer.zero_grad()
            total_loss.backward()
            base.model_optimizer.step()

            meter.update(
                {
                    "global_pid_loss": global_pid_loss.data,
                    "multiview_pid_loss": multiview_pid_loss.data,
                    "contrast_loss": contrast_loss.data,
                }
            )

    return meter
