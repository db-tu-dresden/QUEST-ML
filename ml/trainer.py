import os
import sys
from enum import Enum
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import optim, nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from ml import ddp
from ml.config import Config
from ml.data import ProcessDataset
from ml.logger import Logger
from ml.models import DistributedDataParallel
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

        self.load()

    def save_checkpoint(self, valid_loss: float, valid_accuracy: float):
        if not ddp.is_main_process() or valid_loss >= self.checkpoint_stats['min_valid_loss']:
            return

        os.makedirs(self.config['save_dir'], exist_ok=True)
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'valid_loss': valid_loss,
            'valid_accuracy': valid_accuracy,
        }, self.config['checkpoint_save_path'])

        self.model.save(self.config)

    def load_checkpoint(self):
        if not self.config['load']:
            return
        checkpoint = torch.load(self.config['checkpoint_load_path'])
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

    @staticmethod
    def get_accuracy(outputs: [torch.Tensor], targets: [torch.Tensor]) -> torch.Tensor:
        return (outputs.round() == targets).flatten(start_dim=1).all(dim=1).float().mean()

    @staticmethod
    def get_kl_divergence(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        out_dist = F.log_softmax(outputs, dim=-1)
        target_dist = F.softmax(targets, dim=-1)
        return F.kl_div(out_dist, target_dist, reduction='batchmean')

    def batch_data_to_device(self, batch: Iterable) -> tuple:
        return tuple(elem.to(self.device, non_blocking=True) if torch.is_tensor(elem) else elem for elem in batch)

    def _train_epoch(self, mode: Mode, dataloader: DataLoader) -> (float, float):
        data_size = len(dataloader.dataset) / self.config['world_size']
        num_batches = int(data_size / self.config['batch_size'])

        epoch_loss = torch.tensor(0., device=self.config['device'])
        epoch_accuracy = torch.tensor(0., device=self.config['device'])
        epoch_kl = torch.tensor(0., device=self.config['device'])
        epoch_kl_rounded = torch.tensor(0., device=self.config['device'])

        if mode == Mode.TRAIN:
            self.model.train()
        else:
            self.model.eval()

        inputs, outputs, targets = None, None, None

        with optional(mode != Mode.TRAIN, torch.no_grad):
            for batch_data in tqdm(dataloader, disable=not self.config['verbose'], file=sys.stdout):
                inputs, targets = self.batch_data_to_device(batch_data)

                if mode == Mode.TRAIN:
                    self.optimizer.zero_grad(set_to_none=self.config['set_gradients_none'])

                with optional(self.config['fp16'] and self.scaler, torch.cuda.amp.autocast):
                    outputs = self.model(inputs)
                batch_loss = self.criterion(outputs, targets)
                batch_accuracy = self.get_accuracy(outputs.detach(), targets.detach())

                if mode == Mode.TRAIN:
                    if self.scaler:
                        self.scaler.scale(batch_loss).backward()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        batch_loss.backward()
                        self.optimizer.step()

                    self.logger.log_batch_loss(mode, batch_loss.item())

                if mode == Mode.VALID:
                    epoch_kl += self.get_kl_divergence(outputs.detach(), targets.detach())
                    epoch_kl_rounded += self.get_kl_divergence(outputs.detach().round(), targets.detach())

                epoch_loss += batch_loss
                epoch_accuracy += batch_accuracy

            epoch_loss /= num_batches
            epoch_accuracy /= num_batches
            epoch_kl /= num_batches
            epoch_kl_rounded /= num_batches

            if mode == Mode.VALID and inputs is not None and outputs is not None and targets is not None:
                self.logger.log_data(inputs.detach().cpu(), outputs.detach().round().cpu(), targets.detach().cpu(),
                                     epoch_loss.item(), epoch_accuracy.item())

                self.logger.log_metric(mode.VALID, 'kl_divergence', epoch_kl.item(), min)
                self.logger.log_metric(mode.VALID, 'kl_divergence_rounded', epoch_kl_rounded.item(), min)

        return epoch_loss.item(), epoch_accuracy.item()

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
            self.logger.log(self.model, verbose=False)
            if self.config['wandb_watch_model']:
                self.logger.watch(self.model, self.criterion, log='all')
            for epoch in range(self.config['epochs']):
                self.logger.epoch(epoch)
                if self.train_sampler and self.valid_sampler:
                    self.train_sampler.set_epoch(epoch)
                    self.valid_sampler.set_epoch(epoch)

                train_loss, train_acc = self._train()
                self.logger.log_metric(Mode.TRAIN, 'loss', train_loss, min, to_wandb=False)
                self.logger.log_metric(Mode.TRAIN, 'accuracy', train_acc, max, to_wandb=False)

                valid_loss, valid_acc = self.valid()
                self.save_checkpoint(valid_loss, valid_acc)

                self.logger.log({'learning_rate': self.optimizer.param_groups[0]['lr']})
                self.lr_scheduler.step(valid_loss)

            test_loss, test_acc = self.test()

            self.save()
            self.cleanup()

    def valid(self):
        valid_loss, valid_acc = self._valid()
        self.logger.log_metric(Mode.VALID, 'loss', valid_loss, min)
        self.logger.log_metric(Mode.VALID, 'accuracy', valid_acc, max)
        return valid_loss, valid_acc

    def test(self):
        test_loss, test_acc = self._test()
        self.logger.log_metric(Mode.TEST, 'loss', test_loss, min)
        self.logger.log_metric(Mode.TEST, 'accuracy', test_acc, max)
        return test_loss, test_acc

    def save(self):
        if not ddp.is_main_process():
            return
        self.logger.log_system_config()
        self.logger.log_graph_description()
        self.logger.log_model()

    def load(self):
        self.load_checkpoint()
        self.model.load(self.config)

    @staticmethod
    def cleanup():
        ddp.cleanup()

    @staticmethod
    def get_datasets_from_path(path: str, scaling_factor: int = 1, reduction_factor: float = 0.0, offset: int = 1,
                               only_process: bool = False, pickle_file_name: str = 'da.pkl'):
        return ProcessDataset.from_path(os.path.join(path, 'train', pickle_file_name),
                                        scaling_factor, reduction_factor, offset, only_process), \
            ProcessDataset.from_path(os.path.join(path, 'valid', pickle_file_name),
                                     scaling_factor, reduction_factor, offset, only_process), \
            ProcessDataset.from_path(os.path.join(path, 'test', pickle_file_name),
                                     scaling_factor, reduction_factor, offset, only_process)

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
                                                                           config['reduction_factor'],
                                                                           config['offset'],
                                                                           config['only_process'],
                                                                           config['pickle_file_name'])

        if config['gpu']:
            ddp.run(cls._run, config, model, train_data, valid_data, test_data)
        else:
            cls._run(None, config, model, train_data, valid_data, test_data)
