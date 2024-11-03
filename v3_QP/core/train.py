from network.processing import (
    FeatureMapIntegrating,
    FeatureMapLocalizedIntegratingNoRelu,
    FeatureMapQuantifiedIntegratingProbLogSoftmaxWeights,
)
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

            features_map = base.model(imgs)
            bn_features, cls_score = base.classifier(features_map)
            quantified_features_map, quantified_integrating_features_map, integrating_pids = FeatureMapQuantifiedIntegratingProbLogSoftmaxWeights(config).__call__(features_map, cls_score, pids)

            quantified_integrating_bn_features, quantified_integrating_cls_score = base.classifier2(quantified_integrating_features_map)

            ide_loss = base.pid_creiteron(cls_score, pids)
            quantified_integrating_ide_loss = base.pid_creiteron(quantified_integrating_cls_score, integrating_pids)
            quantified_integrating_reasoning_loss = base.reasoning_creiteron(bn_features, quantified_integrating_bn_features)

            total_loss = ide_loss + quantified_integrating_ide_loss + config.lambda1 * quantified_integrating_reasoning_loss

            base.model_optimizer.zero_grad()
            base.classifier_optimizer.zero_grad()
            base.classifier2_optimizer.zero_grad()
            total_loss.backward()
            base.model_optimizer.step()
            base.classifier_optimizer.step()
            base.classifier2_optimizer.step()

            meter.update(
                {
                    "pid_loss": ide_loss.data,
                    "quantified_integrating_pid_loss": quantified_integrating_ide_loss.data,
                    "quantified_integrating_reasoning_loss": quantified_integrating_reasoning_loss.data,
                }
            )

    return meter.get_val(), meter.get_str()