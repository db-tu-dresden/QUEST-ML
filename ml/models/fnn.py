import torch
from torch import nn

from ml.models.base import Model


class FNN(Model):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, layers: int = 2):
        super().__init__()
        self.layers = [nn.Linear(input_size, hidden_size)] + \
                      [nn.Linear(hidden_size, hidden_size) for _ in range(layers - 1)]
        self.out_layer = nn.Linear(hidden_size, output_size)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor):
        shape = x.shape
        out = x.view(shape[0], -1)
        for layer in self.layers:
            out = self.activation(layer(out))
        out = self.out_layer(out)
        out = out.view(shape[0], shape[1], -1)
        return out


class MLP(Model):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, hidden_layers: int = 2):
        super().__init__()

        self.model = nn.Sequential()
        self.model.add_module('dense1', nn.Linear(input_size, hidden_size))
        self.model.add_module('act1', nn.ReLU())

        for i in range(hidden_layers):
            self.model.add_module(f'dense{i + 2}', nn.Linear(hidden_size, hidden_size))
            self.model.add_module(f'act{i + 2}', nn.ReLU())

        self.model.add_module(f'dense{hidden_layers + 2}',  nn.Linear(hidden_size, output_size))

    def forward(self, x: torch.Tensor):
        shape = x.shape
        out = x.view(shape[0], -1)
        out = self.model(out)
        out = out.view(shape[0], shape[1], -1)
        return out
