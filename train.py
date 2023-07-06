import argparse
import math
import os

from ml import Config, ProcessDataset, Trainer
from ml.models.fnn import FNN

parser = argparse.ArgumentParser(description='ML model for sequence to sequence translation')
parser.add_argument('-p', '--path', help='Path where a config.yaml describing the system and '
                                         'a graph_description.note describing the process graph lie.')


def get_datasets(path: str):
    return ProcessDataset.from_path(os.path.join(path, 'train')), \
        ProcessDataset.from_path(os.path.join(path, 'valid')), \
        ProcessDataset.from_path(os.path.join(path, 'test'))


def run(args):
    base_path = args.path
    config = Config('ml/config.yaml')

    train_ds, valid_ds, test_ds = get_datasets(base_path)

    input_size = math.prod(test_ds.get_sample_shape())  # num processes * num jobs
    hidden_size = config['hidden_size']
    output_size = input_size

    model = FNN(input_size, hidden_size, output_size)

    trainer = Trainer.initialize(config, model, train_ds, valid_ds, test_ds)
    trainer.train()


if __name__ == '__main__':
    args = parser.parse_args()
    run(args)
