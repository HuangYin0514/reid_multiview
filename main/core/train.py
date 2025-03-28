import torch
from method import innovation, loss_function
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
            # D: Decoupling
            second_branch_features = base.model.module.second_branch_gap(features_map).squeeze()
            shared_features, special_features, reconstructed_features = base.model.module.featureDecouplingModule(second_branch_features)

            # F: Fusion
            # fusion_features, fusion_pids = base.model.module.featureFusionModule(shared_features, special_features, pids)

            second_branch_features = torch.cat((shared_features, special_features), dim=1)
            fusion_features, fusion_pids = base.model.module.featureViewsFusionModule(second_branch_features, pids)

            second_branch_fusion_bn_features, second_branch_fusion_fusion_cls_score = base.model.module.second_branch_classifier(second_branch_features)
            second_branch_fusion_pid_loss = loss_function.CrossEntropyLabelSmooth().forward(second_branch_fusion_fusion_cls_score, pids)

            fusion_bn_features, fusion_cls_score = base.model.module.second_branch_classifier(fusion_features)
            fusion_pid_loss = loss_function.CrossEntropyLabelSmooth().forward(fusion_cls_score, fusion_pids)

            fusion_pid_loss = fusion_pid_loss + second_branch_fusion_pid_loss

            #################################################################
            # C: ContrastLoss
            contrast_loss = innovation.multi_view.ContrastLoss(config).__call__(backbone_bn_features, fusion_bn_features, pids)

            #################################################################
            # Total loss
            total_loss = pid_loss + fusion_pid_loss + contrast_loss

            base.model_optimizer.zero_grad()
            total_loss.backward()
            base.model_optimizer.step()

            meter.update(
                {
                    "pid_loss": pid_loss.data,
                    "fusion_pid_loss": fusion_pid_loss.data,
                    "contrast_loss": contrast_loss.data,
                }
            )

    return meter
