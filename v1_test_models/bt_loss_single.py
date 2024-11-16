import torch
import torch.distributed as dist
import torch.nn as nn


class ExampleModel(nn.Module):
    def __init__(self, batch_size, lambd):
        super(ExampleModel, self).__init__()
        self.bn = nn.BatchNorm1d(128)  # 128 features
        self.args = type("", (), {})()  # Create a dummy object to hold arguments
        self.args.batch_size = batch_size
        self.args.lambd = lambd

    def off_diagonal(self, x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def bt_loss_single(self, z1, z2):
        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)

        # sum the cross-correlation matrix between all gpus (simulate distributed)
        c.div_(self.args.batch_size * 4)
        if dist.is_initialized():
            dist.all_reduce(c)
            print(c.shape)

        # On-diagonal loss: (c_{ii} - 1)^2
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()

        # Off-diagonal loss: sum(c_{ij}^2 for i != j)
        off_diag = self.off_diagonal(c).pow_(2).sum()

        # Total loss
        loss = on_diag + self.args.lambd * off_diag
        return loss, on_diag, off_diag


# Simulate a batch of data
batch_size = 8
z1 = torch.randn(batch_size, 128)  # A batch of 8 samples with 128 features
z2 = torch.randn(batch_size, 128)  # Another batch of 8 samples with 128 features

# Initialize model
lambd = 0.5
model = ExampleModel(batch_size, lambd)

# Compute loss
loss, on_diag, off_diag = model.bt_loss_single(z1, z2)

# Output the results
print(f"Loss: {loss.item()}")
print(f"On-diagonal loss: {on_diag.item()}")
print(f"Off-diagonal loss: {off_diag.item()}")
