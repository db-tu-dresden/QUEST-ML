from contextlib import contextmanager
from enum import Enum

import torch
from torch.utils.data import DataLoader

from ml import ddp
from ml.config import Config
from ml.logger import Logger


class Mode(Enum):
    TRAIN = 'train'
    VALID = 'valid'
    TEST = 'test'


@contextmanager
def optional(condition, context_manager):
    if condition:
        with context_manager():
            yield
    else:
        yield


class Trainer:
    def __init__(self, config: Config):
        self.logger = Logger()

        self.config = config
        self.device = None

        self.model = None

        self.scaler = None

        self.train_sampler = None
        self.valid_sampler = None
        self.test_sampler = None

        self.train_dataloader = None
        self.valid_dataloader = None
        self.test_dataloader = None

        self.criterion = None
        self.optimizer = None

        self.scheduler = None

    def batch_data_to_device(self, batch):
        return tuple(elem.to(self.device, non_blocking=True) if torch.is_tensor(elem) else elem for elem in batch)

    def _go(self, mode: Mode, dataloader: DataLoader):
        epoch_loss, epoch_accuracies = torch.zeros(1), torch.zeros(1)

        if mode == Mode.TRAIN:
            self.model.train()
            self.optimizer.zero_grad(set_to_none=self.config['set_gradients_none'])
        else:
            self.model.eval()

        with optional(mode != Mode.TRAIN, torch.no_grad):
            for batch, batch_data in enumerate(dataloader):
                inputs, targets = self.batch_data_to_device(batch_data)
                with optional(self.config['fp16'] and self.scaler, torch.cuda.amp.autocast):
                    decoders_outputs, outputs_seqs = self.model(inputs)
                batch_loss = self.criterion(decoders_outputs, targets)

                if mode == Mode.TRAIN:
                    if self.scaler:
                        self.scaler.scale(batch_loss).backward()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        batch_loss.backward()
                        self.optimizer.step()

                # log step
            # log epoch

        return epoch_loss, epoch_accuracies

    def _train(self):
        return self._go(Mode.TRAIN, self.train_dataloader)

    def _valid(self):
        return self._go(Mode.VALID, self.valid_dataloader)

    def _test(self):
        return self._go(Mode.TEST, self.test_dataloader)

    def cleanup(self):
        ddp.cleanup()

    def train(self):
        for epoch in range(self.config['epochs']):
            self.logger.epoch(epoch)
            if self.train_sampler and self.valid_sampler:
                self.train_sampler.set_epoch(epoch)
                self.valid_sampler.set_epoch(epoch)

            train_loss, train_accuracy = self._train()
            valid_loss, valid_accuracy = self._valid()

            self.scheduler.step()
        test_loss, test_accuracy = self._test()

        self.cleanup()

    def valid(self):
        self._valid()

    def test(self):
        self._test()
