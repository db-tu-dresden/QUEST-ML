import logging

import torch
import wandb
from torch import nn

from ml.config import Config


class Logger:
    def __init__(self, project: str = 'carQUEST-ML'):
        self.wandb = wandb.login()
        self.project = project
        self._epoch = 0
        self._step = 0

    def epoch(self, epoch: int):
        self._epoch = epoch
        self.log(f'-------------------------\n'
                 f'Epoch {epoch + 1}\n'
                 f'-------------------------\n')

    def log_batch(self, loss: torch.Tensor, step: int = None):
        if step is None:
            step = self._step
            self._step += 1

        wandb.log({"epoch": self._epoch, "loss": loss}, step=step)

    def log(self, msg, level=logging.INFO):
        if isinstance(msg, dict):
            if self.wandb:
                wandb.log(msg)
        logging.log(level, msg)

    def watch(self, model: nn.Module, *args, **kwargs):
        wandb.watch(model, *args, **kwargs)

    def __call__(self, config: Config):
        yield wandb.init(config.data, project=self.project)
