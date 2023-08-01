import argparse

import torch
import torch.nn.functional as F
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
    def add_args(parser: argparse.ArgumentParser, prefix: str = ''):
        parser.add_argument(f'--{prefix}input_size', type=int, metavar='N', help='Input size')
        parser.add_argument(f'--{prefix}hidden_size', type=int, metavar='N', help='Hidden size')
        parser.add_argument(f'--{prefix}output_size', type=int, metavar='N', help='Output size')
        parser.add_argument(f'--{prefix}hidden_layers', type=int, metavar='N', help='Number of hidden layers')

    @classmethod
    def build_model(cls, cfg: Config, prefix: str = ''):
        return cls(cfg[f'{prefix}input_size'], cfg[f'{prefix}hidden_size'], cfg[f'{prefix}output_size'],
                   cfg[f'{prefix}hidden_layers'])

    def forward(self, x: torch.Tensor):
        shape = x.shape
        out = x.view(shape[0], -1)
        out = self.model(out)
        out = out.view(*shape)
        return out


@register_model('embedding_mlp')
class EmbeddingMLP(Model):
    def __init__(self, input_size: int, embed_size: int, hidden_size: int, output_size: int,
                 hidden_layers: int = 2, dropout: float = 0.5):
        super().__init__()

        self.input_size = input_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers
        self.dropout = dropout

        self.embedding_in = nn.Linear(self.input_size, self.embed_size)
        self.mlp = MLP(self.embed_size, self.hidden_size, self.embed_size, self.hidden_layers)
        self.embedding_out = nn.Linear(self.embed_size, self.output_size)

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, x: torch.Tensor):                                     # NxPxI
        out = x
        out = self.embedding_in(out)                                        # NxPxE
        out = self.activation(out)
        out = self.dropout(out)

        size = out.size()
        kernel = (out.size(1), 1)

        out, indices = F.max_pool2d(out, kernel, return_indices=True)       # (Nx1xE, Nx1xE)
        out = self.mlp(out)                                                 # Nx1xE
        out = self.dropout(out)
        out = F.max_unpool2d(out, indices, kernel, output_size=size[1:])    # NxPxE
        out = self.embedding_out(out)                                       # NxPxO
        return out

    @staticmethod
    def add_args(parser: argparse.ArgumentParser, prefix: str = ''):
        parser.add_argument(f'--{prefix}input_size', type=int, metavar='N', help='Input size')
        parser.add_argument(f'--{prefix}embedding_size', type=int, metavar='N', help='Embedding size')
        parser.add_argument(f'--{prefix}hidden_size', type=int, metavar='N', help='Hidden size')
        parser.add_argument(f'--{prefix}output_size', type=int, metavar='N', help='Output size')
        parser.add_argument(f'--{prefix}hidden_layers', type=int, metavar='N', help='Number of hidden layers')
        parser.add_argument(f'--{prefix}dropout', type=float, metavar='N', help='Dropout value')

    @classmethod
    def build_model(cls, cfg: Config, prefix: str = ''):
        return cls(cfg['input_size'], cfg['embedding_size'], cfg['hidden_size'], cfg['output_size'],
                   cfg['hidden_layers'], cfg['dropout'])


@register_model_architecture('mlp', 'mlp')
def mlp(cfg: Config, prefix: str = ''):
    cfg['input_size'] = cfg['jobs'] * cfg['processes'] if 'jobs' in cfg and 'processes' in cfg else 16
    cfg['hidden_size'] = cfg['hidden_size'] if 'hidden_size' in cfg else 32
    cfg['output_size'] = cfg['output_size'] if 'output_size' in cfg else cfg['input_size']
    cfg['hidden_layers'] = cfg['hidden_layers'] if 'hidden_layers' in cfg else 2

    cfg[f'{prefix}input_size'] = cfg[f'{prefix}input_size'] if f'{prefix}input_size' in cfg else cfg['input_size']
    cfg[f'{prefix}hidden_size'] = cfg[f'{prefix}hidden_size'] if f'{prefix}hidden_size' in cfg else cfg['hidden_size']
    cfg[f'{prefix}output_size'] = cfg[f'{prefix}output_size'] if f'{prefix}output_size' in cfg else cfg['output_size']
    cfg[f'{prefix}hidden_layers'] = cfg[f'{prefix}hidden_layers'] if f'{prefix}hidden_layers' in cfg else cfg['hidden_layers']


@register_model_architecture('embedding_mlp', 'embedding_mlp')
def embedding_mlp(cfg: Config):
    cfg['input_size'] = cfg['jobs'] if 'jobs' in cfg else 16
    cfg['embedding_size'] = cfg['embedding_size'] if 'embedding_size' in cfg else 16
    cfg['hidden_size'] = cfg['hidden_size'] if 'hidden_size' in cfg else 32
    cfg['output_size'] = cfg['output_size'] if 'output_size' in cfg else cfg['input_size']
    cfg['hidden_layers'] = cfg['hidden_layers'] if 'hidden_layers' in cfg else 2
    cfg['dropout'] = cfg['dropout'] if 'dropout' in cfg else 0.5
