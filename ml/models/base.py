import argparse

import torch
from torch import nn

from ml import ddp, Config


class Model(nn.Module):

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        pass

    @classmethod
    def build_model(cls, cfg: Config):
        pass

    def save(self, path: str):
        if ddp.is_main_process():
            torch.save(self.state_dict(), path)
