import logging

import wandb


class Logger:
    def __init__(self):
        self.wandb = wandb.login()

    def epoch(self, epoch):
        self.log(f'-------------------------\n'
                 f'Epoch {epoch + 1}\n'
                 f'-------------------------\n')

    def log(self, msg, level=logging.INFO):
        if isinstance(msg, dict):
            if self.wandb:
                wandb.log(msg)
        logging.log(level, msg)

    def __call__(self, *args, **kwargs):
        return self.log(*args, **kwargs)
