import torch
from torch import nn

from ml.models.base import Model
from ml.models.mlp import MLP


class MLPEncoder(Model):
    def __init__(self, n_processes: int, input_size: int, embed_size: int, hidden_size: int, output_size: int,
                 dropout: float):
        super().__init__()

        self.n_processes = n_processes
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embed_size = embed_size

        self.embedding = nn.Linear(self.input_size, self.embed_size)

        self.process_mlps = nn.ModuleList([MLP(self.embed_size, self.hidden_size, self.hidden_size)
                                           for _ in range(self.n_processes)])

        self.fusion = MLP(self.hidden_size, self.hidden_size, self.output_size)
        self.activation = nn.Sigmoid()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):     # NxPx3
        out = x
        out = self.embedding(out)           # NxPxE

        out = self.dropout(out)

        res = []
        for i, l in enumerate(self.process_mlps):
            inter = self.process_mlps[i](out[::, i])
            inter = self.dropout(inter)
            res.append(inter)
        out = torch.stack(res)              # NxPxH
        out = torch.sum(out, dim=0)         # NxH

        out = self.dropout(out)

        out = self.fusion(torch.tanh(out))  # NxO
        out = self.activation(out)
        return out


class MLPDecoder(Model):
    def __init__(self, n_processes: int, input_size: int, hidden_size: int, output_size: int, dropout: float):
        super().__init__()

        self.n_processes = n_processes
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.mlp = MLP(self.input_size, self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(dropout)

        self.process_mlps = nn.ModuleList([MLP(self.hidden_size, self.hidden_size, self.output_size)
                                           for _ in range(self.n_processes)])

    def forward(self, x: torch.Tensor):
        out = x                     # NxI
        out = self.mlp(out)         # NxH
        out = self.dropout(out)

        res = []
        for i, l in enumerate(self.process_mlps):
            inter = self.process_mlps[i](out)
            inter = self.dropout(inter)
            res.append(inter)
        out = torch.stack(res, dim=1)  # NxPxO

        return out


class MLPEncoderDecoder(Model):
    def __init__(self, n_processes: int, input_size: int, embed_size: int, hidden_size: int, output_size: int,
                 dropout: float):
        super().__init__()

        self.encoder = MLPEncoder(n_processes, input_size, embed_size, hidden_size, output_size, dropout)
        self.decoder = MLPDecoder(n_processes, output_size, hidden_size, input_size, dropout)

    def forward(self, x: torch.Tensor):
        out = x
        out = self.encoder(out)
        out = self.decoder(out)
        return out
