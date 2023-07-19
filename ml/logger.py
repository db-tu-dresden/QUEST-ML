from __future__ import annotations

import wandb
from torch import nn

from ml.config import Config
from ml.utils import optional

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ml.trainer import Mode


class Logger:
    def __init__(self, config: Config):
        self.config = config
        self.wandb = wandb.login() if self.config['wandb'] else None

        self.float_formatter = '.4f'

        self._epoch = 0
        self._step = 0

    def epoch(self, epoch: int):
        self._epoch = epoch
        self.log(f'-------------------------\n'
                 f'Epoch {epoch + 1} / {self.config["epochs"]}\n'
                 f'-------------------------\n')
        if self.config['wandb']:
            wandb.log({'epoch': self._epoch})

    def log_batch(self, mode: Mode, loss: float, step: int = None):
        if step is None:
            step = self._step
            self._step += 1

        if self.config['wandb']:
            wandb.log({f'{mode.value}_loss': loss}, step=step)

    def log(self, msg, to_wandb: bool = None, verbose: bool = None):
        if verbose is None:
            verbose = self.config['verbose']
        if to_wandb is None:
            to_wandb = self.config['wandb']
        if isinstance(msg, dict):
            if to_wandb:
                wandb.log(msg)
            if verbose:
                print(' | '.join([f'{k}: {v:{self.float_formatter}}' for k, v in msg.items()]))
        else:
            if verbose:
                print(msg)

    def watch(self, model: nn.Module, *args, **kwargs):
        if self.config['wandb']:
            wandb.watch(model, *args, **kwargs)

    def __call__(self, config: Config):
        return optional(self.config['wandb'], wandb.init, config=config.data, project=self.config['wandb_project'],
                        group=self.config['wandb_group'] if self.config['wandb_group'] else None)
