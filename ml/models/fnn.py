import torch
from torch import nn

from ml.models.base import Model


class FNN(Model):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, layers: int = 2):
        super().__init__()
        self.layers = [nn.Linear(input_size, hidden_size)] + \
                      [nn.Linear(hidden_size, hidden_size) for _ in range(layers - 1)]
        self.out_layer = nn.Linear(hidden_size, output_size)
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        out = x
        for layer in self.layers:
            out = self.activation(layer(out))
        out = self.out_layer(out)
        return out


class FNN2(Model):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, hidden_layers: int = 2):
        super().__init__()

        self.model = nn.Sequential()
        self.model.add_module('dense1', nn.Linear(input_size, hidden_size))
        self.model.add_module('act1', nn.ReLU())

        for i in range(hidden_layers):
            self.model.add_module(f'dense{i + 2}', nn.Linear(hidden_size, hidden_size))
            self.model.add_module(f'act{i + 2}', nn.ReLU())

        self.model.add_module(f'dense{hidden_layers + 2}',  nn.Linear(hidden_size, output_size))
        self.model.add_module(f'act{hidden_layers + 2}', nn.Sigmoid())

    def forward(self, x: torch.Tensor):
        return self.model(x)
