import os
import sys
from enum import Enum

import torch
import torch.distributed as dist
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from ml import ddp
from ml.config import Config
from ml.data import ProcessDataset
from ml.logger import Logger
from ml.utils import optional


class Mode(Enum):
    TRAIN = 'train'
    VALID = 'valid'
    TEST = 'test'


class Trainer:
    def __init__(self, config: Config, model,
                 train_data: ProcessDataset, valid_data: ProcessDataset, test_data: ProcessDataset):
        self.logger = Logger(config)

        self.config = config
        self.device = self.config['device']

        self.checkpoint_stats = {
            'min_valid_loss': float('inf')
        }

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
                                           shuffle=self.config['shuffle'],
                                           sampler=self.train_sampler,
                                           num_workers=self.config['num_workers_dataloader'],
                                           pin_memory=self.config['pin_memory'],
                                           drop_last=self.config['drop_last'])
        self.valid_dataloader = DataLoader(self.valid_data, batch_size=self.config['batch_size'],
                                           shuffle=self.config['shuffle'],
                                           sampler=self.valid_sampler,
                                           num_workers=self.config['num_workers_dataloader'],
                                           pin_memory=self.config['pin_memory'],
                                           drop_last=self.config['drop_last'])
        self.test_dataloader = DataLoader(self.test_data, batch_size=self.config['batch_size'],
                                          shuffle=self.config['shuffle'],
                                          sampler=self.test_sampler,
                                          num_workers=self.config['num_workers_dataloader'],
                                          pin_memory=self.config['pin_memory'],
                                          drop_last=self.config['drop_last'])

        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.config['learning_rate'],
                                   momentum=self.config['momentum'])
        self.lr_scheduler = ReduceLROnPlateau(self.optimizer, 'min',
                                              factor=config['lr_scheduler_factor'],
                                              patience=config['lr_scheduler_patience'])

        self.load_checkpoint()

    def checkpoint(self, valid_loss: float, file_name: str = None):
        if ddp.is_main_process():
            os.makedirs(self.config['checkpoint_path'], exist_ok=True)
            torch.save({
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'valid_loss': valid_loss,
            }, os.path.join(self.config['checkpoint_path'], file_name or self.config['checkpoint_file']))

            if valid_loss < self.checkpoint_stats['min_valid_loss']:
                self.checkpoint_stats['min_valid_loss'] = valid_loss
                self.checkpoint(valid_loss, 'best.pt')

    def load_checkpoint(self):
        try:
            checkpoint = torch.load(os.path.join(self.config['checkpoint_path'], self.config['checkpoint_file']))
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        except FileNotFoundError:
            pass

    def batch_data_to_device(self, batch):
        return tuple(elem.to(self.device, non_blocking=True) if torch.is_tensor(elem) else elem for elem in batch)

    def _go(self, mode: Mode, dataloader: DataLoader):
        data_size = len(dataloader.dataset) / self.config['world_size']
        num_batches = int(data_size / self.config['batch_size'])

        epoch_loss = 0

        if mode == Mode.TRAIN:
            self.model.train()
        else:
            self.model.eval()

        with optional(mode != Mode.TRAIN, torch.no_grad):
            for batch_data in tqdm(dataloader, disable=not self.config['verbose'], file=sys.stdout):
                inputs, targets = self.batch_data_to_device(batch_data)

                if mode == Mode.TRAIN:
                    self.optimizer.zero_grad(set_to_none=self.config['set_gradients_none'])

                with optional(self.config['fp16'] and self.scaler, torch.cuda.amp.autocast):
                    outputs = self.model(inputs)
                batch_loss = self.criterion(outputs, targets)

                if mode == Mode.TRAIN:
                    if self.scaler:
                        self.scaler.scale(batch_loss).backward()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        batch_loss.backward()
                        self.optimizer.step()

                    self.logger.log_batch(mode, batch_loss.item())

                epoch_loss += batch_loss.item()

        return epoch_loss / num_batches

    def _train(self):
        return self._go(Mode.TRAIN, self.train_dataloader)

    def _valid(self):
        return self._go(Mode.VALID, self.valid_dataloader)

    def _test(self):
        return self._go(Mode.TEST, self.test_dataloader)

    def train(self, config: dict = None):
        if config is not None:
            self.config.update(config)

        with self.logger(self.config):
            if self.config['wandb_watch_model']:
                self.logger.watch(self.model, self.criterion, log="all")
            for epoch in range(self.config['epochs']):
                self.logger.epoch(epoch)
                if self.train_sampler and self.valid_sampler:
                    self.train_sampler.set_epoch(epoch)
                    self.valid_sampler.set_epoch(epoch)

                train_loss = self._train()
                self.logger.log({'train_loss': train_loss}, to_wandb=False)

                valid_loss = self.valid()
                self.checkpoint(valid_loss)

                self.logger.log({'learning_rate': self.optimizer.param_groups[0]['lr']})
                self.lr_scheduler.step(valid_loss)

            test_loss = self.test()

        if self.config['save_model']:
            self.model.save(path=self.config['model_save_path'])
        self.cleanup()

    def valid(self):
        valid_loss = self._valid()
        self.logger.log({'valid/valid_loss': valid_loss})
        return valid_loss

    def test(self):
        test_loss = self._test()
        self.logger.log({'test/test_loss': test_loss})
        return test_loss

    @staticmethod
    def cleanup():
        ddp.cleanup()

    @staticmethod
    def get_datasets_from_path(path: str, scaling_factor: int = 1, pickle_file_name: str = 'da.pkl'):
        return ProcessDataset.from_path(os.path.join(path, 'train', pickle_file_name), scaling_factor), \
            ProcessDataset.from_path(os.path.join(path, 'valid', pickle_file_name), scaling_factor), \
            ProcessDataset.from_path(os.path.join(path, 'test', pickle_file_name), scaling_factor)

    @classmethod
    def _run(cls, rank: int | None, config: Config, model,
             train_data: ProcessDataset, valid_data: ProcessDataset, test_data: ProcessDataset):
        if rank is not None:
            ddp.setup(rank, config)

        config['world_size'] = dist.get_world_size() if dist.is_initialized() else 1

        trainer = cls(config, model, train_data, valid_data, test_data)
        trainer.train()

    @classmethod
    def run(cls, config: Config, model, train_data: ProcessDataset = None, valid_data: ProcessDataset = None,
            test_data: ProcessDataset = None):
        config['world_size'] = torch.cuda.device_count()
        config['master_port'] = ddp.find_free_port(config['master_addr'])

        if not train_data and not valid_data and not test_data:
            train_data, valid_data, test_data = cls.get_datasets_from_path(config['data_path'],
                                                                           config['scaling_factor'],
                                                                           config['pickle_file_name'])

        if config['on_gpu']:
            ddp.run(cls._run, config, model, train_data, valid_data, test_data)
        else:
            cls._run(None, config, model, train_data, valid_data, test_data)
