from network.processing import FeatureMapLocalizedIntegratingNoRelu
from tools import MultiItemAverageMeter
from tqdm import tqdm


def train(base, loaders, config):

    base.set_train()
    loader = loaders.loader
    meter = MultiItemAverageMeter()
    for i, data in enumerate(tqdm(loader)):
        imgs, pids, cids = data
        imgs, pids, cids = imgs.to(base.device), pids.to(base.device).long(), cids.to(base.device).long()
        if config.module == "B":
            features_map = base.model(imgs)
            bn_features, cls_score = base.classifier(features_map)

            ide_loss = base.pid_creiteron(cls_score, pids)

            total_loss = ide_loss

            base.model_optimizer.zero_grad()
            base.classifier_optimizer.zero_grad()
            total_loss.backward()
            base.model_optimizer.step()
            base.classifier_optimizer.step()

            meter.update(
                {
                    "pid_loss": ide_loss.data,
                }
            )

        elif config.module == "Lucky":

            features_map, hierarchical_score_list, hierarchical_features_map = base.model(imgs)
            bn_features, cls_score = base.classifier(features_map)

            localized_features_map, localized_integrating_features_map, integrating_pids = FeatureMapLocalizedIntegratingNoRelu(config).__call__(hierarchical_features_map, pids, base)
            localized_integrating_bn_features, localized_integrating_cls_score = base.classifier2(localized_integrating_features_map)

            ide_loss = base.pid_creiteron(cls_score, pids)
            localized_integrating_ide_loss = base.pid_creiteron(localized_integrating_cls_score, integrating_pids)
            localized_integrating_reasoning_loss = base.reasoning_creiteron(bn_features, localized_integrating_bn_features)

            # Hierarchical
            other_loss = 0
            for temp_score in hierarchical_score_list:
                other_loss += base.pid_creiteron(temp_score, pids)

            total_loss = ide_loss + localized_integrating_ide_loss + config.lambda1 * localized_integrating_reasoning_loss + other_loss

            base.model_optimizer.zero_grad()
            base.classifier_optimizer.zero_grad()
            base.classifier2_optimizer.zero_grad()
            base.auxiliaryModel_optimizer.zero_grad()
            total_loss.backward()
            base.model_optimizer.step()
            base.classifier_optimizer.step()
            base.classifier2_optimizer.step()
            base.auxiliaryModel_optimizer.step()

            meter.update(
                {
                    "pid_loss": ide_loss.data,
                    "localized_integrating_pid_loss": localized_integrating_ide_loss.data,
                    "localized_integrating_reasoning_loss": localized_integrating_reasoning_loss.data,
                    "other_loss": other_loss,
                }
            )

    return meter.get_val(), meter.get_str()
