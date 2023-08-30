import argparse

import torch
from torch import nn

from ml import Config, ddp


class Model(nn.Module):

    @staticmethod
    def add_args(parser: argparse.ArgumentParser, prefix: str = ''):
        pass

    @classmethod
    def build_model(cls, cfg: Config, prefix: str = ''):
        pass

    def save(self, config: Config):
        if not ddp.is_main_process() or not config['save_model']:
            return
        torch.save({
            'model': self.state_dict()
        }, config['model_save_path'])

    def load(self, config: Config):
        if not config['load_model']:
            return
        checkpoint = torch.load(config['model_load_path'])
        self.load_state_dict(checkpoint['model'])


class DistributedDataParallel(nn.parallel.DistributedDataParallel, Model):
    def save(self, config: Config):
        if not ddp.is_main_process():
            return
        self.module.save(config)

    def load(self, config: Config):
        self.module.load(config)
