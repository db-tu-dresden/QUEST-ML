import argparse
import os

import ml
from ml import Config, ProcessDataset, Trainer
from ml.models import build_model

parser = argparse.ArgumentParser(description='ML model for sequence to sequence translation')
parser.add_argument('-p', '--path', help='Path where a config.yaml describing the system and '
                                         'a graph_description.note describing the process graph lie.')
parser.add_argument('-g', '--gpu', action='store_true', help='Enable GPU usage')
parser.add_argument('-n', '--n_gpus', type=int, metavar='N', help='How many GPUs to use.If not set, all GPUs are used')
parser.add_argument('--wandb', type=bool, default=True, help='Whether to use wandb is used for logging.')
parser.add_argument('--arch', type=str, help='Model architecture')


def get_datasets(path: str, scaling_factor: int):
    return ProcessDataset.from_path(os.path.join(path, 'data', 'train', 'da.pkl'), scaling_factor), \
        ProcessDataset.from_path(os.path.join(path, 'data', 'valid', 'da.pkl'), scaling_factor), \
        ProcessDataset.from_path(os.path.join(path, 'data', 'test', 'da.pkl'), scaling_factor)


def run(args):
    base_path = args.path
    config = Config('ml/config.yaml')
    config.set_base_path(base_path)
    config.set_from_args(args)

    _, _, test_ds = get_datasets(base_path, config['scaling_factor'])

    config['processes'] = test_ds.get_sample_shape()[0]
    config['jobs'] = test_ds.get_sample_shape()[1]

    args.input_size = config['processes'] * config['jobs']
    args.output_size = args.input_size

    model = build_model(args)

    Trainer.run(config, model)


if __name__ == '__main__':
    args = ml.models.parse_arch(parser)
    run(args)
