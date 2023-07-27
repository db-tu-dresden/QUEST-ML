import argparse

import torch
from torch import nn

from ml import Config
from ml.models import register_model, register_model_architecture
from ml.models.base import Model


@register_model('mlp')
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

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        parser.add_argument('--input_size', type=int, metavar='N', help='Input size')
        parser.add_argument('--hidden_size', type=int, metavar='N', help='Hidden size')
        parser.add_argument('--output_size', type=int, metavar='N', help='Output size')
        parser.add_argument('--hidden_layers', type=int, metavar='N', help='Number of hidden layers')

    @classmethod
    def build_model(cls, cfg: Config):
        return cls(cfg['input_size'], cfg['hidden_size'], cfg['output_size'], cfg['hidden_layers'])

    def forward(self, x: torch.Tensor):
        shape = x.shape
        out = x.view(shape[0], -1)
        out = self.model(out)
        out = out.view(*shape)
        return out


@register_model_architecture('mlp', 'mlp')
def mlp(cfg: Config):
    cfg['input_size'] = cfg['input_size'] if 'input_size' in cfg else 16
    cfg['hidden_size'] = cfg['hidden_size'] if 'hidden_size' in cfg else 32
    cfg['output_size'] = cfg['output_size'] if 'output_size' in cfg else cfg['input_size']
    cfg['hidden_layers'] = cfg['hidden_layers'] if 'hidden_layers' in cfg else 2
