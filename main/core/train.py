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

            # ------------- Hard content branch -----------------------
            ## Global
            hard_global_embedding_features = base.model.module.hard_global_embedding(hard_features)
            hard_global_pooling_features = base.model.module.hard_global_pooling(hard_global_embedding_features).squeeze()
            hard_global_bn_features, hard_global_cls_score = base.model.module.hard_global_classifier(hard_global_pooling_features)
            hard_global_pid_loss = loss_function.CrossEntropyLabelSmooth().forward(hard_global_cls_score, pids)
            hard_global_triplet_loss = loss_function.TripletLoss()(hard_global_pooling_features, pids)[0]
            hard_global_loss = hard_global_pid_loss + hard_global_triplet_loss

            ## Parts
            PART_NUM = 2
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

            # ------------- Soft content branch -----------------------
            # Upstream
            soft_upstream_global_features = base.model.module.soft_upstream_global_embedding(soft_features_l4)
            soft_upstream_global_pooling_features = base.model.module.soft_upstream_global_pooling(soft_upstream_global_features).squeeze()
            soft_upstream_global_bn_features, soft_upstream_global_cls_score = base.model.module.soft_upstream_global_classifier(soft_upstream_global_pooling_features)
            soft_upstream_global_pid_loss = loss_function.CrossEntropyLabelSmooth().forward(soft_upstream_global_cls_score, pids)
            soft_upstream_global_triplet_loss = loss_function.TripletLoss()(soft_upstream_global_pooling_features, pids)[0]
            soft_upstream_global_loss = soft_upstream_global_pid_loss + soft_upstream_global_triplet_loss

            # Upstream attention
            soft_upstream_attention_attentions, soft_upstream_attention_bap_AiF_features, soft_upstream_attention_bap_features = base.model.module.soft_upstream_attention(soft_features_l4)
            soft_upstream_attention_bn_features, soft_upstream_attention_cls_score = base.model.module.soft_upstream_attention_classifier(soft_upstream_attention_bap_features)
            soft_upstream_attention_pid_loss = loss_function.CrossEntropyLabelSmooth().forward(soft_upstream_attention_cls_score, pids)
            soft_upstream_attention_loss = soft_upstream_attention_pid_loss

            # Downstream
            soft_downstream_l4_embedding_features = base.model.module.soft_downstream_l4_embedding(soft_features_l3)
            soft_downstream_global_embedding_features = base.model.module.soft_downstream_global_embedding(soft_downstream_l4_embedding_features)
            soft_downstream_global_pooling_features = base.model.module.soft_downstream_global_pooling(soft_downstream_global_embedding_features).squeeze()
            soft_downstream_global_bn_features, soft_downstream_global_cls_score = base.model.module.soft_downstream_global_classifier(soft_downstream_global_pooling_features)
            soft_downstream_global_pid_loss = loss_function.CrossEntropyLabelSmooth().forward(soft_downstream_global_cls_score, pids)
            soft_downstream_global_triplet_loss = loss_function.TripletLoss()(soft_downstream_global_pooling_features, pids)[0]
            soft_downstream_global_loss = soft_downstream_global_pid_loss + soft_downstream_global_triplet_loss

            # Downstream attention
            soft_downstream_attention_attentions, soft_downstream_attention_bap_AiF_features, soft_downstream_attention_bap_features = base.model.module.guide_dualscale_attention(soft_features_l3, soft_downstream_l4_embedding_features, soft_upstream_attention_attentions)
            soft_downstream_attention_bn_features, soft_downstream_attention_cls_score = base.model.module.soft_downstream_attention_classifier(soft_downstream_attention_bap_features)
            soft_downstream_attention_pid_loss = loss_function.CrossEntropyLabelSmooth().forward(soft_downstream_attention_cls_score, pids)
            soft_downstream_attention_loss = soft_downstream_attention_pid_loss

            # ------------- Fusion content branch -----------------------
            fusion_features = base.model.module.fusion(soft_upstream_global_features, hard_part_embedding_features_list, soft_upstream_global_features, soft_downstream_global_embedding_features)
            fusion_pooling_features = base.model.module.fusion_pooling(fusion_features).squeeze()
            fusion_bn_features, fusion_cls_score = base.model.module.fusion_classifier(fusion_pooling_features)
            fusion_pid_loss = loss_function.CrossEntropyLabelSmooth().forward(fusion_cls_score, pids)
            fusion_triplet_loss = loss_function.TripletLoss()(fusion_pooling_features, pids)[0]
            fusion_loss = fusion_pid_loss + fusion_triplet_loss

            #################################################################
            # Total loss
            total_loss = hard_global_loss + hard_part_loss + soft_upstream_global_loss + soft_downstream_global_loss + soft_upstream_attention_loss + soft_downstream_attention_loss + fusion_loss

            base.model_optimizer.zero_grad()
            total_loss.backward()
            base.model_optimizer.step()

            meter.update(
                {
                    "hard_global_loss": hard_global_loss.data,
                    "hard_part_loss": hard_part_loss.data,
                    "soft_upstream_global_loss": soft_upstream_global_loss.data,
                    "soft_downstream_global_loss": soft_downstream_global_loss.data,
                    "soft_upstream_attention_loss": soft_upstream_attention_loss.data,
                    "soft_downstream_attention_loss": soft_downstream_attention_loss.data,
                }
            )

    return meter
