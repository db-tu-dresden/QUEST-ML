import argparse
import logging
import os

import yaml
from schema import Schema, And, Use, SchemaError, Or

from system.config import Config as SystemConfig


def validate_yaml(data: dict, schema: Schema):
    try:
        return schema.validate(data)
    except SchemaError as e:
        logging.exception(e)


def true(*args, **kwargs):
    return True


def greater_zero(x):
    return x > 0


def greater_equal_zero(x):
    return x > 0


def between_zero_one(x):
    return 0 <= x <= 1


def length_greater_zero(x):
    return len(x) > 0


class Config:
    config_dict = {
        # general
        'gpu': {'type': bool, 'lambda': true},
        'world_size': {'type': int, 'lambda': greater_equal_zero},
        'job_id': {'type': int, 'lambda': true},

        # logging
        'wandb': {'type': bool, 'lambda': true},
        'wandb_project': {'type': str, 'lambda': true},
        'wandb_group': {'type': str, 'lambda': true},
        'wandb_watch_model': {'type': bool, 'lambda': true},
        'wandb_table_name': {'type': str, 'lambda': true},
        'wandb_table_elements': {'type': int, 'lambda': greater_zero},
        'verbose': {'type': bool, 'lambda': true},
        'float_precision': {'type': int, 'lambda': greater_zero},
        'log_to_file': {'type': bool, 'lambda': true},

        # training hyperparameters
        'epochs': {'type': int, 'lambda': greater_zero},
        'learning_rate': {'type': float, 'lambda': greater_zero},
        'momentum': {'type': float, 'lambda': greater_zero},
        'batch_size': {'type': int, 'lambda': greater_zero},
        'lambda': {'type': float, 'lambda': between_zero_one},

        # learning rate scheduler
        'lr_scheduler_factor': {'type': float, 'lambda': greater_zero},
        'lr_scheduler_patience': {'type': int, 'lambda': greater_zero},

        # training configuration
        'stepwise': {'type': bool, 'lambda': true},
        'set_gradients_none': {'type': bool, 'lambda': true},
        'fp16': {'type': bool, 'lambda': true},
        'allow_tf32': {'type': bool, 'lambda': true},

        # dataloader parameters
        'shuffle': {'type': bool, 'lambda': true},
        'drop_last': {'type': bool, 'lambda': true},
        'pin_memory': {'type': bool, 'lambda': true},
        'num_workers_dataloader': {'type': int, 'lambda': true},

        # data
        'pickle_file_name': {'type': str, 'lambda': length_greater_zero},
        'processes': {'type': int, 'lambda': true},
        'jobs': {'type': int, 'lambda': true},
        'scaling_factor': {'type': int, 'lambda': greater_zero},
        'reduction_factor': {'type': float, 'lambda': between_zero_one},
        'offset': {'type': int, 'lambda': greater_zero},
        'only_process': {'type': bool, 'lambda': true},

        # distributed training
        'master_addr': {'type': str, 'lambda': length_greater_zero, 'can_be_none': True},
        'master_port': {'type': int, 'lambda': true, 'can_be_none': True},
        'device': {'type': int, 'lambda': true, 'can_be_none': True},

        # model checkpoint
        'min_checkpoint_epoch': {'type': int, 'lambda': true},
        'min_checkpoint_epoch_dist': {'type': int, 'lambda': true},

        # save/load
        'save': {'type': bool, 'lambda': true},
        'load': {'type': bool, 'lambda': true},
        'save_model': {'type': bool, 'lambda': true},
        'load_model': {'type': bool, 'lambda': true},

        # paths
        'base_path': {'type': str, 'lambda': true},
        'data_path': {'type': str, 'lambda': true},
        'save_dir': {'type': str, 'lambda': true},
        'checkpoint_file': {'type': str, 'lambda': true},
        'checkpoint_save_path': {'type': str, 'lambda': true},
        'checkpoint_load_path': {'type': str, 'lambda': true},
        'model_file': {'type': str, 'lambda': true},
        'model_save_path': {'type': str, 'lambda': true},
        'model_load_path': {'type': str, 'lambda': true},
        'system_config_path': {'type': str, 'lambda': true},
        'graph_description_path': {'type': str, 'lambda': true},
        'output_file': {'type': str, 'lambda': true},
    }

    default_schema = Schema({
        k:
            Or(None, And(Use(v['type']), v['lambda'])) if 'can_be_none' in v and v['can_be_none'] else
            And(Use(v['type']), v['lambda'])
        for k, v in config_dict.items()
    })

    def __init__(self, path: str, schema: Schema = default_schema):
        self.path = path
        self.schema = schema

        with open(self.path) as f:
            self.data = yaml.full_load(f)

        self.validate()

    def set_base_path(self, base_path: str = None):
        if base_path is None:
            base_path = self['base_path']
        self.data['base_path'] = base_path
        self.data['data_path'] = self.data['data_path'] or os.path.join(base_path, 'data')
        self.data['save_dir'] = self.data['save_dir'] or os.path.join(base_path, 'save')

        self.data['model_save_path'] = (self.data['model_save_path'] or
                                        os.path.join(self.data['save_dir'], self.data['model_file']))
        self.data['model_load_path'] = (self.data['model_save_path'] or
                                        os.path.join(self.data['save_dir'], self.data['model_file']))

        self.data['checkpoint_save_path'] = (self.data['checkpoint_save_path'] or
                                             os.path.join(self.data['save_dir'], self.data['checkpoint_file']))
        self.data['checkpoint_load_path'] = (self.data['checkpoint_load_path'] or
                                             os.path.join(self.data['save_dir'], self.data['checkpoint_file']))

        self.data['system_config_path'] = (self.data['system_config_path'] or
                                           os.path.join(base_path, 'config.yaml'))
        self.data['graph_description_path'] = (self.data['graph_description_path'] or
                                               os.path.join(base_path, 'graph_description.note'))

        self.data['output_file'] = (self.data['output_file'] or
                                    os.path.join(base_path, 'job-' + str(self.data['job_id']) + '.out'))

    def set_system_data(self):
        sc = SystemConfig(self['system_config_path'])
        self.data['processes'] = len(sc['processes']) + 1
        self.data['jobs'] = len(sc['jobs'])

    def update_from_args(self, args: argparse.Namespace):
        d = vars(args)
        for k, v in d.items():
            self.data[k] = v

        self.set_base_path()
        self.set_system_data()

    def validate(self):
        self.data = validate_yaml(self.data, self.schema)

    def update(self, new_config: dict):
        self.data.update(new_config)

    def __repr__(self):
        cls = self.__class__.__name__
        return f'{cls}(path={self.path!r}, schema={self.schema!r})'

    def __str__(self):
        return str({
            'data': self.data,
            'schema': self.schema,
        })

    def __getitem__(self, item):
        return self.data[item]

    def __contains__(self, item):
        return item in self.data

    def __setitem__(self, key, value):
        self.data[key] = value
