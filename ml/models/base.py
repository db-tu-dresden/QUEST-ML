import torch
from torch import nn

from ml import ddp


class Model(nn.Module):

    @staticmethod
    def add_args(parser):
        pass

    @classmethod
    def build_model(cls, args):
        pass

    def save(self, path: str):
        if ddp.is_main_process():
            torch.save(self.state_dict(), path)
