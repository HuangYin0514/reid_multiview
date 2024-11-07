import torch.nn as nn


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_out")
        nn.init.constant_(m.bias, 0.0)
    elif classname.find("Conv") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find("BatchNorm") != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
    elif classname.find("InstanceNorm") != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class BaseClassifier(nn.Module):
    def __init__(self, hindden_dim, pid_num):
        super(BaseClassifier, self).__init__()

        self.pid_num = pid_num
        self.GAP = nn.AdaptiveAvgPool2d((1, 1))
        self.BN = nn.BatchNorm1d(hindden_dim)
        self.BN.apply(weights_init_kaiming)

        self.classifier = nn.Linear(hindden_dim, self.pid_num, bias=False)
        self.classifier.apply(weights_init_classifier)

    def forward(self, features_map):
        features = self.GAP(features_map)
        bn_features = self.BN(features.squeeze())
        cls_score = self.classifier(bn_features)
        return bn_features, cls_score


class TransLayer_classifier(nn.Module):
    def __init__(self, config):
        super(TransLayer_classifier, self).__init__()

        self.num_layer = 3

        input_channel = [256, 256, 256, 256]
        classifier_list = nn.ModuleList()
        for i in range(self.num_layer):
            temp = BaseClassifier(input_channel[i], config.pid_num)
            classifier_list.append(temp)
        self.classifier_list = classifier_list

    def forward(self, xs):
        assert isinstance(xs, (tuple, list))
        assert len(xs) == 3

        score_list = []
        for i in range(self.num_layer):
            _, score = self.classifier_list[i](xs[i])
            score_list.append(score)

        return score_list


class TransLayer_1(nn.Module):
    def __init__(self):
        super().__init__()

        self.num_layer = 3

        kernel_size = [(4, 4), (2, 2), (1, 1), (1, 1)]
        pool_list = nn.ModuleList()
        for i in range(self.num_layer):
            temp = nn.MaxPool2d(kernel_size=kernel_size[i])
            pool_list.append(temp)
        self.pool_list = pool_list

        input_channel = [256, 512, 1024, 2048]
        output_channel = [256, 256, 256, 256]
        cv1_list = nn.ModuleList()
        for i in range(self.num_layer):
            temp = nn.Sequential(
                nn.Conv2d(input_channel[i], output_channel[i], kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(output_channel[i]),
                nn.ReLU(inplace=True),
            )
            cv1_list.append(temp)
        self.cv1_list = cv1_list

    def forward(self, xs):
        assert isinstance(xs, (tuple, list))
        assert len(xs) == 3

        cv_feats_list = []
        for i in range(self.num_layer):
            pool_feats = self.pool_list[i](xs[i])
            cv_feats = self.cv1_list[i](pool_feats)
            cv_feats_list.append(cv_feats)

        return cv_feats_list
