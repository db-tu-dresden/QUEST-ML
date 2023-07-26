import os
import sys
from enum import Enum

import torch
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

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
                                           shuffle=None if self.train_sampler else self.config['shuffle'],
                                           sampler=self.train_sampler,
                                           num_workers=self.config['num_workers_dataloader'],
                                           pin_memory=self.config['pin_memory'],
                                           drop_last=self.config['drop_last'])
        self.valid_dataloader = DataLoader(self.valid_data, batch_size=self.config['batch_size'],
                                           shuffle=None if self.valid_sampler else self.config['shuffle'],
                                           sampler=self.valid_sampler,
                                           num_workers=self.config['num_workers_dataloader'],
                                           pin_memory=self.config['pin_memory'],
                                           drop_last=self.config['drop_last'])
        self.test_dataloader = DataLoader(self.test_data, batch_size=self.config['batch_size'],
                                          shuffle=None if self.test_sampler else self.config['shuffle'],
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

    def checkpoint(self, valid_loss: float, valid_accuracy: float, file_name: str = None):
        if ddp.is_main_process():
            os.makedirs(self.config['checkpoint_path'], exist_ok=True)
            torch.save({
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'valid_loss': valid_loss,
                'valid_accuracy': valid_accuracy,
            }, os.path.join(self.config['checkpoint_path'], file_name or self.config['checkpoint_file']))

            if valid_loss < self.checkpoint_stats['min_valid_loss']:
                self.checkpoint_stats['min_valid_loss'] = valid_loss
                self.checkpoint(valid_loss, valid_accuracy, 'best.pt')

    def load_checkpoint(self):
        if not self.config['from_checkpoint']:
            return
        try:
            checkpoint = torch.load(os.path.join(self.config['checkpoint_path'], self.config['checkpoint_file']))
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        except FileNotFoundError:
            pass

    @staticmethod
    def get_accuracy(outputs: [torch.Tensor], targets: [torch.Tensor]):
        return (outputs.round() == targets).all(axis=2).all(axis=1).sum().item() / outputs.shape[0]

    def batch_data_to_device(self, batch):
        return tuple(elem.to(self.device, non_blocking=True) if torch.is_tensor(elem) else elem for elem in batch)

    def _train_epoch(self, mode: Mode, dataloader: DataLoader):
        data_size = len(dataloader.dataset) / self.config['world_size']
        num_batches = int(data_size / self.config['batch_size'])

        epoch_loss = 0
        epoch_accuracy = 0

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
                batch_accuracy = self.get_accuracy(outputs, targets)

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
                epoch_accuracy += batch_accuracy

            epoch_loss /= num_batches
            epoch_accuracy /= num_batches

            if mode == Mode.VALID:
                self.logger.log_data(inputs.detach().cpu(), outputs.detach().round().cpu(), targets.detach().cpu(),
                                     epoch_loss, epoch_accuracy)

        return epoch_loss, epoch_accuracy

    def _train(self):
        return self._train_epoch(Mode.TRAIN, self.train_dataloader)

    def _valid(self):
        return self._train_epoch(Mode.VALID, self.valid_dataloader)

    def _test(self):
        return self._train_epoch(Mode.TEST, self.test_dataloader)

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

                train_loss, train_acc = self._train()
                self.logger.log({
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                }, to_wandb=False)

                valid_loss, valid_acc = self.valid()
                self.checkpoint(valid_loss, valid_acc)

                self.logger.log({'learning_rate': self.optimizer.param_groups[0]['lr']})
                self.lr_scheduler.step(valid_loss)

            test_loss, test_acc = self.test()

        if self.config['save_model']:
            self.model.save(path=self.config['model_save_path'])
        self.cleanup()

    def valid(self):
        valid_loss, valid_acc = self._valid()
        self.logger.log({
            'valid/valid_loss': valid_loss,
            'valid/valid_accuracy': valid_acc,
        })
        return valid_loss, valid_acc

    def test(self):
        test_loss, test_acc = self._test()
        self.logger.log({
            'test/test_loss': test_loss,
            'test/test_accuracy': test_acc,
        })
        return test_loss, test_acc

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

        trainer = cls(config, model, train_data, valid_data, test_data)
        trainer.train()

    @classmethod
    def run(cls, config: Config, model, train_data: ProcessDataset = None, valid_data: ProcessDataset = None,
            test_data: ProcessDataset = None):
        config['world_size'] = config['world_size'] or torch.cuda.device_count() or 1
        config['master_port'] = ddp.find_free_port(config['master_addr'])

        if not train_data and not valid_data and not test_data:
            train_data, valid_data, test_data = cls.get_datasets_from_path(config['data_path'],
                                                                           config['scaling_factor'],
                                                                           config['pickle_file_name'])

        if config['gpu']:
            ddp.run(cls._run, config, model, train_data, valid_data, test_data)
        else:
            cls._run(None, config, model, train_data, valid_data, test_data)
