import torch
from torch import nn

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
    def add_args(parser):
        parser.add_argument('--input_size', type=int, metavar='N', help='Input size')
        parser.add_argument('--hidden_size', type=int, metavar='N', help='Hidden size')
        parser.add_argument('--output_size', type=int, metavar='N', help='Output size')
        parser.add_argument('--hidden_layers', type=int, metavar='N', help='Number of hidden layers')

    @classmethod
    def build_model(cls, args):
        return cls(args.input_size, args.hidden_size, args.output_size, args.hidden_layers)

    def forward(self, x: torch.Tensor):
        shape = x.shape
        out = x.view(shape[0], -1)
        out = self.model(out)
        out = out.view(shape[0], shape[1], -1)
        return out


@register_model_architecture('mlp', 'mlp')
def mlp(args):
    args.input_size = getattr(args, 'input_size', 16)
    args.hidden_size = getattr(args, 'hidden_size', 32)
    args.output_size = getattr(args, 'output_size', args.input_size)
    args.hidden_layers = getattr(args, 'hidden_layers', 2)
