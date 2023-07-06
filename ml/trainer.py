from contextlib import contextmanager
from enum import Enum

import torch
import torch.distributed as dist
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from ml import ddp
from ml.config import Config
from ml.data import ProcessDataset
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
    def __init__(self, config: Config, model,
                 train_data: ProcessDataset, valid_data: ProcessDataset, test_data: ProcessDataset):
        self.logger = Logger()

        self.config = config
        self.device = self.config['device']

        self.model = model

        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data

        self.scaler = None
        self.train_sampler = None
        self.valid_sampler = None
        self.test_sampler = None

        if ddp.is_dist_avail_and_initialized():
            self.scaler = torch.cuda.amp.GradScaler()

            self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_data)
            self.valid_sampler = torch.utils.data.distributed.DistributedSampler(self.valid_data)
            self.test_sampler = torch.utils.data.distributed.DistributedSampler(self.test_data)

            self.model = DistributedDataParallel(self.model.to(self.device), device_ids=[self.config['rank']])

        self.train_dataloader = DataLoader(self.train_data, batch_size=self.config['batch_size'],
                                           sampler=self.train_sampler,
                                           num_workers=self.config['num_workers_dataloader'],
                                           pin_memory=self.config['pin_memory'])
        self.valid_dataloader = DataLoader(self.valid_data, batch_size=self.config['batch_size'],
                                           sampler=self.valid_sampler,
                                           num_workers=self.config['num_workers_dataloader'],
                                           pin_memory=self.config['pin_memory'])
        self.test_dataloader = DataLoader(self.test_data, batch_size=self.config['batch_size'],
                                          sampler=self.test_sampler,
                                          num_workers=self.config['num_workers_dataloader'],
                                          pin_memory=self.config['pin_memory'])

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.config['learning_rate'],
                                   momentum=self.config['momentum'])

    def batch_data_to_device(self, batch):
        return tuple(elem.to(self.device, non_blocking=True) if torch.is_tensor(elem) else elem for elem in batch)

    def _go(self, mode: Mode, dataloader: DataLoader):
        epoch_loss = torch.zeros(1)

        self.logger.watch(self.model, self.criterion, log="all", log_freq=10)

        if mode == Mode.TRAIN:
            self.model.train()
            self.optimizer.zero_grad(set_to_none=self.config['set_gradients_none'])
        else:
            self.model.eval()

        with optional(mode != Mode.TRAIN, torch.no_grad):
            for batch_data in dataloader:
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

                self.logger.log_batch(batch_loss)
                epoch_loss += batch_loss

        return epoch_loss

    def _train(self):
        return self._go(Mode.TRAIN, self.train_dataloader)

    def _valid(self):
        return self._go(Mode.VALID, self.valid_dataloader)

    def _test(self):
        return self._go(Mode.TEST, self.test_dataloader)

    def train(self):
        with self.logger(self.config):
            for epoch in range(self.config['epochs']):
                self.logger.epoch(epoch)
                if self.train_sampler and self.valid_sampler:
                    self.train_sampler.set_epoch(epoch)
                    self.valid_sampler.set_epoch(epoch)

                train_loss, train_accuracy = self._train()
                valid_loss, valid_accuracy = self._valid()

            test_loss, test_accuracy = self._test()

        self.model.save(path=self.config['model_save_path'])
        self.cleanup()

    def valid(self):
        self._valid()

    def test(self):
        self._test()

    @staticmethod
    def cleanup():
        ddp.cleanup()

    @classmethod
    def _initialize(cls, rank: int | None, config: Config, model,
                    train_data: ProcessDataset, valid_data: ProcessDataset, test_data: ProcessDataset):
        if rank is not None:
            ddp.setup(rank, config)

        config['world_size'] = dist.get_world_size() if dist.is_initialized() else 1

        return cls(config, model, train_data, valid_data, test_data)

    @classmethod
    def initialize(cls, config: Config, model,
                   train_data: ProcessDataset, valid_data: ProcessDataset, test_data: ProcessDataset):
        config['world_size'] = torch.cuda.device_count()
        config['master_port'] = ddp.find_free_port(config['master_addr'])

        if config['on_gpu']:
            ddp.run(cls._initialize, config, model, train_data, valid_data, test_data)
        else:
            return cls._initialize(None, config, model, train_data, valid_data, test_data)
