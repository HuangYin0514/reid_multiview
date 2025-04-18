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
            resnet_feature_maps, copy_resnet_feature_maps = base.model(imgs)

            # ------------- Hard content branch -----------------------
            # Global
            global_features = base.model.module.global_pooling(resnet_feature_maps).squeeze()
            global_bn_features, global_cls_score = base.model.module.global_classifier(global_features)
            global_pid_loss = loss_function.CrossEntropyLabelSmooth().forward(global_cls_score, pids)
            total_loss += global_pid_loss

            # Part
            PART_NUM = config.MODEL.PART_NUM
            hard_part_chunk_features = torch.chunk(resnet_feature_maps, PART_NUM, dim=2)
            hard_part_pid_loss = 0.0
            for i in range(PART_NUM):
                hard_part_chunk_feature_item = hard_part_chunk_features[i]
                hard_part_pooling_features = base.model.module.hard_part_pooling[i](hard_part_chunk_feature_item).squeeze()
                hard_part_bn_features, hard_part_cls_score = base.model.module.hard_part_classifier[i](hard_part_pooling_features)
                hard_part_pid_loss += (1 / PART_NUM) * loss_function.CrossEntropyLabelSmooth().forward(hard_part_cls_score, pids)
            total_loss += hard_part_pid_loss

            # ------------- Soft content branch -----------------------
            # Global
            soft_global_pooling_features = base.model.module.soft_global_pooling(copy_resnet_feature_maps).squeeze()
            soft_global_bn_features, soft_global_cls_score = base.model.module.soft_global_classifier(soft_global_pooling_features)
            soft_global_pid_loss = loss_function.CrossEntropyLabelSmooth().forward(soft_global_cls_score, pids)
            total_loss += soft_global_pid_loss

            # ------------- Multiview content branch  -----------------------
            # Positioning
            multiview_localized_features_map = base.model.module.multiview_feature_map_location(resnet_feature_maps, pids, base.model.module.global_classifier)

            # Quantification
            multiview_localized_features = base.model.module.multiview_pooling(multiview_localized_features_map).squeeze()
            _, multiview_localized_cls_score = base.model.module.global_classifier(multiview_localized_features)
            multiview_quantified_localized_features = base.model.module.multiview_feature_quantification(
                multiview_localized_features,
                multiview_localized_cls_score,
                pids,
            )

            # Fusion
            multiview_fusion_features, multiview_fusion_pids = base.model.module.multiview_feature_fusion(multiview_quantified_localized_features, pids)

            multiview_fusion_bn_features, multiview_cls_score = base.model.module.multiview_classifier(multiview_fusion_features)
            multiview_pid_loss = loss_function.CrossEntropyLabelSmooth().forward(multiview_cls_score, multiview_fusion_pids)
            total_loss += multiview_pid_loss

            # ------------- ContrastLoss  -----------------------
            # contrast_loss = base.model.module.contrast_loss(global_bn_features, multiview_fusion_bn_features)
            contrast_loss = base.model.module.contrast_kl_loss(global_cls_score, multiview_cls_score, global_bn_features, multiview_fusion_bn_features)
            total_loss += contrast_loss

            base.model_optimizer.zero_grad()
            total_loss.backward()
            base.model_optimizer.step()

            meter.update(
                {
                    "global_pid_loss": global_pid_loss.data,
                    "hard_part_pid_loss": hard_part_pid_loss.data,
                    "multiview_pid_loss": multiview_pid_loss.data,
                    "contrast_loss": contrast_loss.data,
                }
            )

    return meter
