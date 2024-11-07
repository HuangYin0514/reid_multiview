import torch
import torch.nn as nn
import torch.nn.functional as F


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

        # 自定义分类器
        input_channel = [256, 256, 256, 256]
        classifier_list = nn.ModuleList()
        for i in range(self.num_layer):
            temp = BaseClassifier(input_channel[i], config.pid_num)
            classifier_list.append(temp)
        self.classifier_list = classifier_list

    def forward(self, feats):
        assert isinstance(feats, (tuple, list))
        assert len(feats) == self.num_layer

        score_list = []
        for i in range(self.num_layer):
            _, score = self.classifier_list[i](feats[i])
            score_list.append(score)

        return score_list


class TransLayer_1(nn.Module):
    def __init__(self):
        super().__init__()

        self.num_layer = 3

        # 池化层
        kernel_size = [(4, 4), (2, 2), (1, 1), (1, 1)]
        pool_list = nn.ModuleList()
        for i in range(self.num_layer):
            temp = nn.MaxPool2d(kernel_size=kernel_size[i])
            pool_list.append(temp)
        self.pool_list = pool_list

        # 1x1卷积层
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

    def forward(self, feats):
        assert isinstance(feats, (tuple, list))
        assert len(feats) == self.num_layer

        outs_list = []
        for i in range(self.num_layer):
            outs = self.pool_list[i](feats[i])
            outs = self.cv1_list[i](outs)
            outs_list.append(outs)

        return outs_list


class Residual(nn.Module):
    def __init__(self, fn):
        super(Residual, self).__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class SEAM(nn.Module):
    def __init__(self, c1, c2, n, reduction=16):
        super(SEAM, self).__init__()
        if c1 != c2:
            c2 = c1
        self.DCovN = nn.Sequential(
            # nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1, groups=c1),
            # nn.GELU(),
            # nn.BatchNorm2d(c2),
            *[
                nn.Sequential(
                    Residual(
                        nn.Sequential(
                            nn.Conv2d(in_channels=c2, out_channels=c2, kernel_size=3, stride=1, padding=1, groups=c2),
                            nn.GELU(),
                            nn.BatchNorm2d(c2),
                        )
                    ),
                    nn.Conv2d(in_channels=c2, out_channels=c2, kernel_size=1, stride=1, padding=0, groups=1),
                    nn.GELU(),
                    nn.BatchNorm2d(c2),
                )
                for i in range(n)
            ]
        )
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(c2, c2 // reduction, bias=False), nn.ReLU(inplace=True), nn.Linear(c2 // reduction, c2, bias=False), nn.Sigmoid())

        self._initialize_weights()
        # self.initialize_layer(self.avg_pool)
        self.initialize_layer(self.fc)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.DCovN(x)
        y = self.avg_pool(y).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y = torch.exp(y)
        return x * y.expand_as(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=1)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def initialize_layer(self, layer):
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            torch.nn.init.normal_(layer.weight, mean=0.0, std=0.001)
            if layer.bias is not None:
                torch.nn.init.constant_(layer.bias, 0)
