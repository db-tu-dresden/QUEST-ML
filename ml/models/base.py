import argparse

import torch
from torch import nn

from ml import Config, ddp


class Model(nn.Module):

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        pass

    @classmethod
    def build_model(cls, cfg: Config):
        pass

    def save(self, config: Config):
        if ddp.is_main_process():
            if config['save_model']:
                torch.save({
                    'model': self.state_dict()
                }, config['model_save_path'])

    def load(self, config: Config):
        if config['load_model']:
            checkpoint = torch.load(config['model_load_path'])
            self.load_state_dict(checkpoint['model'])


class DistributedDataParallel(nn.parallel.DistributedDataParallel, Model):
    pass
