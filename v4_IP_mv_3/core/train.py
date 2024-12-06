from network.common import *
from network.contrastive_loss import *
from network.feature_map_processing import *
from tools import CrossEntropyLabelSmooth, MultiItemAverageMeter
from tqdm import tqdm


def train(base, loaders, config):
    base.set_train()
    loader = loaders.loader
    meter = MultiItemAverageMeter()
    for epoch, data in enumerate(tqdm(loader)):
        imgs, pids, cids = data
        imgs, pids, cids = imgs.to(base.device), pids.to(base.device).long(), cids.to(base.device).long()
        if config.module == "Lucky":
            features_map = base.model(imgs)
            bn_features = base.model.module.gap_bn(features_map)
            shared_features, special_features = base.model.module.decoupling(bn_features)
            bn_features = torch.cat([shared_features, special_features], dim=1)

            # Backbone
            bn_features, cls_score = base.model.module.bn_classifier(bn_features)
            ide_loss = CrossEntropyLabelSmooth().forward(cls_score, pids)

            # # 多视角
            # integrating_features, integrating_pids = base.model.module.feature_integrating(bn_features, pids)
            # integrating_bn_features, integrating_cls_score = base.model.module.bn_classifier2(integrating_features)
            # integrating_ide_loss = CrossEntropyLabelSmooth().forward(integrating_cls_score, integrating_pids)

            # 特征解耦
            bn_shared_features, shared_cls_score = base.model.module.decoupling_shared_bn_classifier(shared_features)
            shared_ide_loss = CrossEntropyLabelSmooth().forward(shared_cls_score, pids)
            bn_special_features, special_cls_score = base.model.module.decoupling_special_bn_classifier(special_features)
            special_ide_loss = CrossEntropyLabelSmooth().forward(special_cls_score, pids)

            num_views = 4
            bs = cls_score.size(0)
            chunk_bs = int(bs / num_views)
            decoupling_loss = 0
            for i in range(chunk_bs):
                shared_feature_i = bn_shared_features[num_views * i : num_views * (i + 1), ...]
                special_feature_i = bn_special_features[num_views * i : num_views * (i + 1), ...]
                # (共享-指定)损失
                sharedSpecialLoss = SharedSpecialLoss().forward(shared_feature_i, special_feature_i)
                # (共享)损失
                sharedSharedLoss = SharedSharedLoss().forward(shared_feature_i)
                # (指定)损失
                # specialSpecialLoss = SpecialSpecialLoss().forward(special_feature_i)
                decoupling_loss += sharedSpecialLoss + 0.1 * sharedSharedLoss

            # 总损失
            total_loss = ide_loss + decoupling_loss + shared_ide_loss + special_ide_loss

            # 反向传播
            base.model_optimizer.zero_grad()
            total_loss.backward()
            base.model_optimizer.step()

            meter.update(
                {
                    "pid_loss": ide_loss.data,
                }
            )

    return meter.get_val(), meter.get_str()
