from network.loss_function import *
from network.processing import *
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
            # ===========================================================
            # Identity Embedding (IDE) Loss Calculation
            # ===========================================================
            features_map = base.model(imgs)
            global_features = base.model.module.decoupling_gap_bn(features_map)  # Global Average Pooling + Batch Norm
            shared_features, specific_features = base.model.module.featureDecoupling(global_features)  # Decoupling features
            reconstructed_features = base.model.module.featureReconstruction(shared_features, specific_features)  # Feature Fusion
            _, classification_scores = base.model.module.classifier(reconstructed_features)  # Final classification features and scores
            ide_loss = CrossEntropyLabelSmooth().forward(classification_scores, pids)

            # ===========================================================
            # Feature Decoupling Loss Calculation
            # ===========================================================
            # 定位
            localized_features_map = FeatureMapLocation(config).__call__(features_map, pids, base.model.module.classifier)
            global_features = base.model.module.decoupling_gap_bn(localized_features_map)
            shared_features, specific_features = base.model.module.featureDecoupling(global_features)

            # Shared feature classification loss
            _, shared_class_scores = base.model.module.decoupling_shared_classifier(shared_features)
            shared_ide_loss = CrossEntropyLabelSmooth().forward(shared_class_scores, pids)

            # Specific feature classification loss
            _, specific_class_scores = base.model.module.decoupling_special_classifier(specific_features)
            specific_ide_loss = CrossEntropyLabelSmooth().forward(specific_class_scores, pids)

            decoupling_loss = DecouplingConsistencyLoss().forward(shared_features, specific_features)

            # regularization_loss = FeatureRegularizationLoss().forward(reconstructed_features)

            # ===========================================================
            # Total Loss Calculation
            # ===========================================================
            total_loss = ide_loss + decoupling_loss + shared_ide_loss + specific_ide_loss

            base.model_optimizer.zero_grad()
            total_loss.backward()
            base.model_optimizer.step()

            meter.update(
                {
                    "pid_loss": ide_loss.data,
                    "decoupling_loss": decoupling_loss.data,
                    "shared_ide_loss": shared_ide_loss.data,
                    "specific_ide_loss": specific_ide_loss.data,
                    "regularization_loss": regularization_loss.data,
                }
            )

    return meter.get_dict(), meter.get_str()
