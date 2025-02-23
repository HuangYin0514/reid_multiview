import torch
import torch.nn.functional as F
from network import (
    CrossEntropyLabelSmooth,
    DecouplingSharedSharedLoss,
    DecouplingSharedSpecialLoss,
    FeatureMapLocation,
    FeatureRegularizationLoss,
    FeatureVectorIntegration,
    FeatureVectorQuantification,
    TripletLoss,
)
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
            global_feat, local_feat_1, local_feat_2, local_feat_3, local_feat_4 = base.model(imgs)
            backbone_bn_features, backbone_cls_score = base.model.module.backbone_classifier(global_feat)
            ide_loss = F.cross_entropy(backbone_cls_score, pids)
            tri_loss = TripletLoss()(global_feat, pids)[0]

            # Local
            _, local_cls_score_1 = base.model.module.local_classifier_1(global_feat)
            ide_loss_local_1 = F.cross_entropy(local_cls_score_1, pids)
            tri_loss_local_1 = TripletLoss()(local_feat_1, pids)[0]

            _, local_cls_score_2 = base.model.module.local_classifier_2(global_feat)
            ide_loss_local_2 = F.cross_entropy(local_cls_score_2, pids)
            tri_loss_local_2 = TripletLoss()(local_feat_2, pids)[0]

            _, local_cls_score_3 = base.model.module.local_classifier_3(global_feat)
            ide_loss_local_3 = F.cross_entropy(local_cls_score_3, pids)
            tri_loss_local_3 = TripletLoss()(local_feat_3, pids)[0]

            _, local_cls_score_4 = base.model.module.local_classifier_4(global_feat)
            ide_loss_local_4 = F.cross_entropy(local_cls_score_4, pids)
            tri_loss_local_4 = TripletLoss()(local_feat_4, pids)[0]

            local_ide_loss = 1 / 4 * (ide_loss_local_1 + ide_loss_local_2 + ide_loss_local_3 + ide_loss_local_4)
            local_tri_loss = 1 / 4 * (tri_loss_local_1 + tri_loss_local_2 + tri_loss_local_3 + tri_loss_local_4)

            #################################################################
            # Loss
            total_loss = 0.5 * ide_loss + 0.5 * local_ide_loss + 0.5 * tri_loss + 0.5 * local_tri_loss

            base.model_optimizer.zero_grad()
            total_loss.backward()
            base.model_optimizer.step()

            meter.update(
                {
                    "pid_loss": ide_loss.data,
                    "tri_loss": tri_loss.data,
                }
            )

    return meter.get_dict(), meter.get_str()
