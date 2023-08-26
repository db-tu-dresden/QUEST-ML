import torch.nn.functional as F
from torch import nn


class CombinedLoss(nn.Module):
    def __init__(self, lamb: float):
        assert 0 <= lamb <= 1
        super().__init__()

        self.lamb = lamb

        self.mse = nn.MSELoss()
        self.kl_div = nn.KLDivLoss(reduction='batchmean', log_target=True)

    def forward(self, inputs, targets):
        in_dists = F.log_softmax(inputs, dim=-1)
        target_dists = F.log_softmax(targets, dim=-1)

        return self.lamb * self.mse(inputs, targets) + (1 - self.lamb) * self.kl_div(in_dists, target_dists)
