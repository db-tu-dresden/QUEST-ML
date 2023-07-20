import argparse
import os

from ml import Config, ProcessDataset, Trainer
from ml.models.fnn import FNN

parser = argparse.ArgumentParser(description='ML model for sequence to sequence translation')
parser.add_argument('-p', '--path', help='Path where a config.yaml describing the system and '
                                         'a graph_description.note describing the process graph lie.')
parser.add_argument('-g', '--gpu', action='store_true', help='Enable GPU usage')


def get_datasets(path: str, scaling_factor: int):
    return ProcessDataset.from_path(os.path.join(path, 'data', 'train', 'da.pkl'), scaling_factor), \
        ProcessDataset.from_path(os.path.join(path, 'data', 'valid', 'da.pkl'), scaling_factor), \
        ProcessDataset.from_path(os.path.join(path, 'data', 'test', 'da.pkl'), scaling_factor)


def run(args):
    base_path = args.path
    config = Config('ml/config.yaml')
    config['base_path'] = base_path
    config['data_path'] = os.path.join(base_path, 'data')
    config['checkpoint_path'] = os.path.join(base_path, 'checkpoint')

    config['on_gpu'] = args.gpu

    _, _, test_ds = get_datasets(base_path, config['scaling_factor'])

    config['processes'] = test_ds.get_sample_shape()[0]
    config['jobs'] = test_ds.get_sample_shape()[1]

    input_size = config['processes'] * config['jobs']  # num processes * num jobs
    hidden_size = config['hidden_size']
    output_size = input_size

    model = FNN(input_size, hidden_size, output_size)

    Trainer.run(config, model)


if __name__ == '__main__':
    args = parser.parse_args()
    run(args)
