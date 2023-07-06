import torch
from torch import nn

from ml.models.base import Model


class FNN(Model):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor):
        out = self.fc1(x)
        out = self.sigmoid(out)
        out = self.fc2(out)
        return out
