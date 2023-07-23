import os

import ml
from ml import Config, ProcessDataset, Trainer, parser
from ml.models import build_model

parser.add_argument('-p', '--path', required=True, help='Path where a config.yaml describing the system and '
                                                        'a graph_description.note describing the process graph lie.')


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

    config['input_size'] = config['processes'] * config['jobs']
    config['output_size'] = config['input_size']

    model = build_model(config)

    Trainer.run(config, model)


if __name__ == '__main__':
    args = ml.models.parse_arch(parser)
    run(args)
