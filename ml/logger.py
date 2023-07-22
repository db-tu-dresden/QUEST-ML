from __future__ import annotations

from copy import copy

import numpy as np
import torch
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

        self.float_formatter = f'.{config["float_precision"]}f'

        self._epoch = 0

        self.data_table = wandb.Table(columns=['epoch', 'loss', 'accuracy', 'inputs', 'outputs', 'targets']) \
            if self.config['wandb'] else None

    def epoch(self, epoch: int):
        self._epoch = epoch
        self.log(f'\n'
                 f'\n'
                 f'-------------------------\n'
                 f'Epoch {epoch + 1} / {self.config["epochs"]}\n'
                 f'-------------------------\n')
        if self.config['wandb'] and wandb.run is not None:
            wandb.log({'epoch': self._epoch})

    def log_batch(self, mode: Mode, loss: float):
        if self.config['wandb'] is None or wandb.run is None:
            return
        wandb.log({f'{mode.value}/{mode.value}_loss': loss})

    def log_data(self, inputs: torch.Tensor, outputs: torch.Tensor, targets: torch.Tensor,
                 loss: float, accuracy: float):
        if self.config['wandb'] is None or wandb.run is None:
            return
        n = self.config['wandb_table_elements']
        self.data_table.add_data(self._epoch, loss, accuracy,
                                 np.array2string(inputs[:n].numpy()),
                                 np.array2string(outputs[:n].numpy()),
                                 np.array2string(targets[:n].numpy()))
        wandb.run.log({self.config['wandb_table_name']: copy(self.data_table)})

    def log(self, msg, to_wandb: bool = None, verbose: bool = None):
        if verbose is None:
            verbose = self.config['verbose']
        if to_wandb is None:
            to_wandb = self.config['wandb']
        if isinstance(msg, dict):
            if to_wandb and wandb.run is not None:
                wandb.log(msg)
            if verbose:
                print(' | '.join([f'{k}: {v:{self.float_formatter}}' for k, v in msg.items()]))
        else:
            if verbose:
                print(msg)

    def watch(self, model: nn.Module, *args, **kwargs):
        if self.config['wandb'] and wandb.run is not None:
            wandb.watch(model, *args, **kwargs)

    def __call__(self, config: Config):
        return optional(self.config['wandb'], wandb.init, config=config.data, project=self.config['wandb_project'],
                        group=self.config['wandb_group'] if self.config['wandb_group'] else None)
