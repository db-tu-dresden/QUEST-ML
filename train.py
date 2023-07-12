import argparse
import math
import os

from ml import Config, ProcessDataset, Trainer
from ml.models.fnn import FNN

parser = argparse.ArgumentParser(description='ML model for sequence to sequence translation')
parser.add_argument('-p', '--path', help='Path where a config.yaml describing the system and '
                                         'a graph_description.note describing the process graph lie.')


def get_datasets(path: str):
    return ProcessDataset.from_path(os.path.join(path, 'data', 'train', 'df.pkl')), \
        ProcessDataset.from_path(os.path.join(path, 'data', 'valid', 'df.pkl')), \
        ProcessDataset.from_path(os.path.join(path, 'data', 'test', 'df.pkl'))


def run(args):
    base_path = args.path
    config = Config('ml/config.yaml')

    _, _, test_ds = get_datasets(base_path)

    input_size = math.prod(test_ds.get_sample_shape())  # num processes * num jobs
    hidden_size = config['hidden_size']
    output_size = input_size

    model = FNN(input_size, hidden_size, output_size)

    trainer = Trainer.initialize(config, model, data_path=base_path)
    trainer.train()


if __name__ == '__main__':
    args = parser.parse_args()
    run(args)
