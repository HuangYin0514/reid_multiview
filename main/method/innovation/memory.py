import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd, nn


class Memory(autograd.Function):
    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())
        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        features = ctx.features
        momentum = ctx.momentum

        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(features)

        # 更新样本记忆
        for x, y in zip(inputs, targets):
            features[y] = momentum * features[y] + (1.0 - momentum) * x
            features[y] /= features[y].norm()

        return grad_inputs, None, None, None  # 对于 targets, features, momentum 无需梯度


class MemoryBank(nn.Module):
    def __init__(self, num_features, num_classes, temp=0.05, momentum=0.01):
        super(MemoryBank, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes

        self.momentum = momentum
        self.temp = temp

        # 初始化样本记忆
        self.features_bank = nn.Parameter(torch.randn(num_classes, num_features))

    def forward(self, inputs, targets, epoch=None):
        norm_inputs = F.normalize(inputs, dim=1)

        # alpha = self.alpha * epoch
        outputs = Memory.apply(norm_inputs, targets, self.features_bank, self.momentum)
        outputs /= self.temp
        loss = F.cross_entropy(outputs, targets)
        return loss


if __name__ == "__main__":
    print("main")
    # 定义测试参数
    num_features = 2048  # 特征维度
    num_classes = 702  # 类别数
    batch_size = 64  # 批量大小

    # 初始化网络
    model_inv = MemoryBank(num_features, num_classes)

    # 创建随机输入数据 (batch_size, num_features)
    inputs = torch.randn(batch_size, num_features, requires_grad=True)

    # 随机生成标签 (batch_size)
    targets = torch.randint(0, num_classes, (batch_size,))

    # 设定 epoch（影响 alpha 的计算）
    epoch = 10

    # 前向传播
    loss = model_inv(inputs, targets, epoch=epoch)

    # 反向传播
    loss.backward()

    # 打印损失
    print("Loss:", loss.item())

    # 打印部分梯度检查
    print("Input gradients:", inputs.grad)
