import math
import os
from enum import Enum

import torch
import torch.distributed as dist
from ray import tune
from ray.air import session
from ray.air.checkpoint import Checkpoint
from ray.tune.schedulers import ASHAScheduler
from tqdm import tqdm
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from ml import ddp
from ml.config import Config
from ml.data import ProcessDataset
from ml.logger import Logger
from ml.models.fnn import FNN
from ml.utils import optional


class Mode(Enum):
    TRAIN = 'train'
    VALID = 'valid'
    TEST = 'test'


class Trainer:
    def __init__(self, config: Config, model,
                 train_data: ProcessDataset, valid_data: ProcessDataset, test_data: ProcessDataset):
        self.logger = Logger(config)
        self.post_epoch_hooks = []

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
        if self.model:
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.config['learning_rate'],
                                       momentum=self.config['momentum'])

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
            for batch_data in tqdm(dataloader, disable=not self.config['verbose']):
                inputs, targets = self.batch_data_to_device(batch_data)
                inputs = inputs.view(self.config['batch_size'], -1)
                targets = targets.view(self.config['batch_size'], -1)

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
                self.logger.log({'train_loss': train_loss})

                valid_loss = self.valid()

                for hook in self.post_epoch_hooks:
                    hook(train_loss, valid_loss)

            test_loss = self.test()

        if self.config['save_model']:
            self.model.save(path=self.config['model_save_path'])
        self.cleanup()

    def valid(self):
        valid_loss = self._valid()
        self.logger.log({'valid_loss': valid_loss})
        return valid_loss

    def test(self):
        test_loss = self._test()
        self.logger.log({'test_loss': test_loss})
        return test_loss

    @staticmethod
    def cleanup():
        ddp.cleanup()

    def _ray_tune_checkpoint(self, train_loss: float, valid_loss: float):
        os.makedirs(self.config['checkpoint_path'], exist_ok=True)
        torch.save(
            (self.model.state_dict(), self.optimizer.state_dict()),
            os.path.join(self.config['checkpoint_path'], 'checkpoint.pt'))
        checkpoint = Checkpoint.from_directory(self.config['checkpoint_path'])
        session.report({'loss': valid_loss}, checkpoint=checkpoint)

    def _ray_tune_train(self, config: dict):
        self.config.update(config)

        self.train_dataloader = DataLoader(self.train_data, batch_size=self.config['batch_size'],
                                           shuffle=self.config['shuffle'],
                                           num_workers=self.config['num_workers_dataloader'],
                                           pin_memory=self.config['pin_memory'],
                                           drop_last=self.config['drop_last'])
        self.valid_dataloader = DataLoader(self.valid_data, batch_size=self.config['batch_size'],
                                           shuffle=self.config['shuffle'],
                                           num_workers=self.config['num_workers_dataloader'],
                                           pin_memory=self.config['pin_memory'],
                                           drop_last=self.config['drop_last'])
        self.test_dataloader = DataLoader(self.test_data, batch_size=self.config['batch_size'],
                                          shuffle=self.config['shuffle'],
                                          num_workers=self.config['num_workers_dataloader'],
                                          pin_memory=self.config['pin_memory'],
                                          drop_last=self.config['drop_last'])

        # build model
        input_size = math.prod(self.train_data.get_sample_shape())
        hidden_size = self.config['hidden_size']
        output_size = input_size
        layers = self.config['layers']
        self.model = FNN(input_size, hidden_size, output_size, layers)

        # move model to device if on gpu
        device = 'cpu'
        if torch.cuda.is_available():
            device = 'cuda:0'
            if torch.cuda.device_count() > 1:
                self.model = nn.DataParallel(self.model)
        self.model.to(device)

        self.optimizer = optim.SGD(self.model.parameters(), lr=self.config['learning_rate'],
                                   momentum=self.config['momentum'])

        # load checkpoint if exists
        loaded_checkpoint = session.get_checkpoint()
        if loaded_checkpoint:
            with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
                model_state, optimizer_state = torch.load(os.path.join(loaded_checkpoint_dir, 'checkpoint.pt'))
            self.model.load_state_dict(model_state)
            self.optimizer.load_state_dict(optimizer_state)

        # train
        self.train()

    def ray_tune(self, tune_config: dict, num_samples: int = 10, max_num_epochs: int = 10, gpus_per_trial: int = 2):
        scheduler = ASHAScheduler(
            max_t=max_num_epochs,
            grace_period=1,
            reduction_factor=2)

        tuner = tune.Tuner(
            tune.with_resources(
                tune.with_parameters(self._ray_tune_train),
                resources={'cpu': 2, 'gpu': gpus_per_trial}
            ),
            tune_config=tune.TuneConfig(
                metric='loss',
                mode='min',
                scheduler=scheduler,
                num_samples=num_samples,
            ),
            param_space=tune_config,
        )

        self.post_epoch_hooks.append(self._ray_tune_checkpoint)
        results = tuner.fit()

        best_result = results.get_best_result('loss', 'min')

        self.logger.log(f'Best trial config: {best_result.config}', verbose=True)
        self.logger.log(f'Best trial final validation loss: {best_result.metrics["loss"]}', verbose=True)
        self.logger.log(f'Best trial final validation accuracy: {best_result.metrics["accuracy"]}', verbose=True)

        # build model
        input_size = math.prod(self.train_data.get_sample_shape())
        hidden_size = best_result.config['hidden_size']
        output_size = input_size
        layers = best_result.config['layers']
        self.model = FNN(input_size, hidden_size, output_size, layers)

        # move model to device if on gpu
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.model.to(device)

        self.test()

    @staticmethod
    def get_datasets_from_path(path: str):
        return ProcessDataset.from_path(os.path.join(path, 'train', 'df.pkl')), \
            ProcessDataset.from_path(os.path.join(path, 'valid', 'df.pkl')), \
            ProcessDataset.from_path(os.path.join(path, 'test', 'df.pkl'))

    @classmethod
    def _initialize(cls, rank: int | None, config: Config, model,
                    train_data: ProcessDataset, valid_data: ProcessDataset, test_data: ProcessDataset):
        if rank is not None:
            ddp.setup(rank, config)

        config['world_size'] = dist.get_world_size() if dist.is_initialized() else 1

        return cls(config, model, train_data, valid_data, test_data)

    @classmethod
    def initialize(cls, config: Config, model, train_data: ProcessDataset = None, valid_data: ProcessDataset = None,
                   test_data: ProcessDataset = None):
        config['world_size'] = torch.cuda.device_count()
        config['master_port'] = ddp.find_free_port(config['master_addr'])

        if not train_data and not valid_data and not test_data:
            train_data, valid_data, test_data = cls.get_datasets_from_path(config['data_path'])

        if config['on_gpu']:
            ddp.run(cls._initialize, config, model, train_data, valid_data, test_data)
        else:
            return cls._initialize(None, config, model, train_data, valid_data, test_data)
