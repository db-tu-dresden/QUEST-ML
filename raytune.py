import argparse
import os.path

import numpy as np
from ray import tune

from ml import Trainer, Config


parser = argparse.ArgumentParser(description='ML model for sequence to sequence translation')
parser.add_argument('-p', '--path', help='Path where a config.yaml describing the system and '
                                         'a graph_description.note describing the process graph lie.')
parser.add_argument('-s', '--samples', help='Number of times to sample from the hyperparameter space.', default=10)
parser.add_argument('-e', '--max-epochs', help='Max number of epochs per trail.', default=10)
parser.add_argument('-g', '--gpus', help='GPUs used per trail.', default=0)


def run(args):
    tune_config = {
        'hidden_size': tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        'layers': tune.sample_from(lambda _: np.random.randint(2, 4)),
        'learning_rate': tune.loguniform(1e-4, 1e-1),
        'batch_size': tune.choice([2, 4, 8, 16])
    }

    base_path = args.path
    trainer_config = Config(os.path.abspath('ml/config.yaml'))
    trainer_config['checkpoint_path'] = os.path.join(base_path, 'checkpoint')
    trainer_config['verbose'] = False

    train_ds, valid_ds, test_ds = Trainer.get_datasets_from_path(base_path)

    trainer = Trainer.initialize(trainer_config, None, train_ds, valid_ds, test_ds)
    trainer.ray_tune(tune_config, args.samples, args.max_epochs, args.gpus)


if __name__ == '__main__':
    args = parser.parse_args()
    run(args)
