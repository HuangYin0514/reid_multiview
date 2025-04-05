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
            #################################################################
            # R: Resnet
            hard_features, soft_features_l3, soft_features_l4 = base.model(imgs)

            # ------------- Hard content branch -----------------------
            ## Global
            hard_global_embedding_features = base.model.module.hard_global_embedding(hard_features)
            hard_global_pooling_features = base.model.module.hard_global_pooling(hard_global_embedding_features).squeeze()
            hard_global_bn_features, hard_global_cls_score = base.model.module.hard_global_classifier(hard_global_pooling_features)
            hard_global_pid_loss = loss_function.CrossEntropyLabelSmooth().forward(hard_global_cls_score, pids)
            hard_global_triplet_loss = loss_function.TripletLoss()(hard_global_pooling_features, pids)[0]
            hard_global_loss = hard_global_pid_loss + hard_global_triplet_loss

            ## Parts
            PART_NUM = config.MODEL.PART_NUM
            hard_part_chunk_features = torch.chunk(hard_features, PART_NUM, dim=2)
            hard_part_embedding_features_list = []
            hard_part_pid_loss = 0.0
            hard_part_triplet_loss = 0.0
            for i in range(PART_NUM):
                hard_part_chunk_feature_item = hard_part_chunk_features[i]
                hard_part_embedding_features = base.model.module.hard_part_embedding[i](hard_part_chunk_feature_item)
                hard_part_embedding_features_list.append(hard_part_embedding_features)
                hard_part_pooling_features = base.model.module.hard_part_pooling[i](hard_part_embedding_features).squeeze()
                hard_part_bn_features, hard_part_cls_score = base.model.module.hard_part_classifier[i](hard_part_pooling_features)
                hard_part_pid_loss += loss_function.CrossEntropyLabelSmooth().forward(hard_part_cls_score, pids)
                hard_part_triplet_loss += loss_function.TripletLoss()(hard_part_pooling_features, pids)[0]
            hard_part_loss = hard_part_pid_loss + hard_part_triplet_loss

            # ------------- Multiview content branch  -----------------------
            # D: Decoupling
            multiview_global_embedding_feature_maps = base.model.module.multiview_global_embedding(soft_features_l4)
            multiview_global_embedding_features = base.model.module.multiview_global_pooling(multiview_global_embedding_feature_maps).squeeze()
            multiview_global_shared_features, multiview_global_specific_features = base.model.module.multiview_global_decoupling(multiview_global_embedding_features)

            # F: Fusion
            multiview_global_shared_fusion_features, integrating_pids = base.model.module.multiview_global_shared_feature_fusion(multiview_global_shared_features, pids)
            multiview_global_specific_fusion_features, integrating_pids = base.model.module.multiview_global_shared_feature_fusion(multiview_global_specific_features, pids)
            multiview_global_fusion_features = torch.cat([multiview_global_shared_fusion_features, multiview_global_specific_fusion_features], dim=1)

            # I: IDLoss
            multiview_global_bn_features, multiview_global_cls_score = base.model.module.multiview_global_classifier(multiview_global_fusion_features)
            multiview_global_pid_loss = loss_function.CrossEntropyLabelSmooth().forward(multiview_global_cls_score, integrating_pids)
            multiview_global_triplet_loss = loss_function.TripletLoss()(multiview_global_fusion_features, integrating_pids)[0]
            multiview_global_loss = multiview_global_pid_loss + multiview_global_triplet_loss

            # ------------- ContrastLoss  -----------------------
            # contrast_loss = innovation.multi_view.ContrastLoss(config).__call__(backbone_bn_features, integrating_bn_features)

            #################################################################
            # Total loss
            total_loss = hard_global_loss + hard_part_loss + multiview_global_loss

            base.model_optimizer.zero_grad()
            total_loss.backward()
            base.model_optimizer.step()

            meter.update(
                {
                    "hard_global_loss": hard_global_loss.data,
                    "hard_part_loss": hard_part_loss.data,
                    "multiview_global_loss": multiview_global_loss.data,
                }
            )

    return meter
