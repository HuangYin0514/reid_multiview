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
            ## Global
            hard_global_embedding_features = base.model.module.hard_global_embedding(hard_features)
            hard_global_pooling_features = base.model.module.hard_global_pooling(hard_global_embedding_features).squeeze()
            hard_global_bn_features, hard_global_cls_score = base.model.module.hard_global_classifier(hard_global_pooling_features)
            hard_global_pid_loss = loss_function.CrossEntropyLabelSmooth().forward(hard_global_cls_score, pids)
            hard_global_triplet_loss = loss_function.TripletLoss()(hard_global_pooling_features, pids)[0]

            ## Parts
            PART_NUM = 2
            hard_part_chunk_features = torch.chunk(hard_features, PART_NUM, dim=2)
            hard_part_pid_loss = 0.0
            hard_part_triplet_loss = 0.0
            for i in range(PART_NUM):
                hard_part_chunk_feature_item = hard_part_chunk_features[i]
                hard_part_embedding_features = base.model.module.hard_part_embedding[i](hard_part_chunk_feature_item)
                hard_part_pooling_features = base.model.module.hard_part_pooling[i](hard_part_embedding_features).squeeze()
                hard_part_bn_features, hard_part_cls_score = base.model.module.hard_part_classifier[i](hard_part_pooling_features)
                hard_part_pid_loss += loss_function.CrossEntropyLabelSmooth().forward(hard_part_cls_score, pids)
                hard_part_triplet_loss += loss_function.TripletLoss()(hard_part_pooling_features, pids)[0]

            #################################################################
            # Total loss
            total_loss = hard_global_pid_loss + hard_part_pid_loss + hard_global_triplet_loss + hard_part_triplet_loss

            base.model_optimizer.zero_grad()
            total_loss.backward()
            base.model_optimizer.step()

            meter.update(
                {
                    "hard_global_pid_loss": hard_global_pid_loss.data,
                    "hard_part_pid_loss": hard_part_pid_loss.data,
                    "hard_global_triplet_loss": hard_global_triplet_loss.data,
                    "hard_part_triplet_loss": hard_part_triplet_loss.data,
                }
            )

    return meter
