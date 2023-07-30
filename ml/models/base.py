import argparse

from torch import nn

from ml import Config


class Model(nn.Module):

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        pass

    @classmethod
    def build_model(cls, cfg: Config):
        pass


class DistributedDataParallel(nn.parallel.DistributedDataParallel, Model):
    pass
