import os

from ml import Config, ProcessDataset, Trainer, Parser
from ml.models import build_model


def get_datasets(path: str, scaling_factor: int):
    return ProcessDataset.from_path(os.path.join(path, 'data', 'train', 'da.pkl'), scaling_factor), \
        ProcessDataset.from_path(os.path.join(path, 'data', 'valid', 'da.pkl'), scaling_factor), \
        ProcessDataset.from_path(os.path.join(path, 'data', 'test', 'da.pkl'), scaling_factor)


def run():
    config = Config('ml/config.yaml')

    parser = Parser(config)
    args = parser.parse_args()

    config.update_from_args(args)

    _, _, test_ds = get_datasets(config['base_path'], config['scaling_factor'])

    config['processes'] = test_ds.get_sample_shape()[0]
    config['jobs'] = test_ds.get_sample_shape()[1]

    config['input_size'] = config['processes'] * config['jobs']
    config['output_size'] = config['input_size']

    model = build_model(config)

    Trainer.run(config, model)


if __name__ == '__main__':
    run()
