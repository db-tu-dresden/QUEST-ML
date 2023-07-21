import argparse
import math
import os.path

import numpy as np
from ray import tune
from ray.air import session
from ray.air.checkpoint import Checkpoint
from ray.tune.schedulers import ASHAScheduler
import torch
from torch import nn
import wandb

from ml import Trainer, Config
from ml.models.fnn import MLP

parser = argparse.ArgumentParser(description='ML model for sequence to sequence translation')
parser.add_argument('-p', '--path', help='Path where a config.yaml describing the system and '
                                         'a graph_description.note describing the process graph lie.')
parser.add_argument('-s', '--samples', help='Number of times to sample from the hyperparameter space.', default=10)
parser.add_argument('-e', '--max-epochs', help='Max number of epochs per trail.', default=10)
parser.add_argument('-g', '--gpus', help='GPUs used per trail.', default=0)


TUNE_CONFIG = {
    'hidden_size': tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
    'layers': tune.sample_from(lambda _: np.random.randint(2, 4)),
    'learning_rate': tune.loguniform(1e-4, 1e-1),
    'batch_size': tune.choice([2, 4, 8, 16])
}


def build_tuner(config: Config, tune_config: dict, num_samples: int = 10, max_num_epochs: int = 10,
                gpus_per_trial: int = 2):
    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train, config=config),
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

    results = tuner.fit()

    best_result = results.get_best_result('loss', 'min')

    print(f'Best trial config: {best_result.config}')
    print(f'Best trial final validation loss: {best_result.metrics["loss"]}')
    print(f'Best trial final validation accuracy: {best_result.metrics["accuracy"]}')

    config.update(best_result.config)

    test(config)


def checkpoint(trainer: Trainer, valid_loss: float, file_name: str = None):
    os.makedirs(trainer.config['checkpoint_path'], exist_ok=True)
    torch.save(
        (trainer.model.state_dict(), trainer.optimizer.state_dict()),
        os.path.join(trainer.config['checkpoint_path'], trainer.config['checkpoint_file']))
    checkpoint = Checkpoint.from_directory(trainer.config['checkpoint_path'])
    session.report({'loss': valid_loss}, checkpoint=checkpoint)


def load_checkpoint(trainer: Trainer):
    # load checkpoint if exists
    loaded_checkpoint = session.get_checkpoint()
    if loaded_checkpoint:
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
            model_state, optimizer_state = torch.load(os.path.join(loaded_checkpoint_dir,
                                                                   trainer.config['checkpoint_file']))
        trainer.model.load_state_dict(model_state)
        trainer.optimizer.load_state_dict(optimizer_state)


def train(tune_config: dict, config: Config):
    config.update(tune_config)

    train_data, valid_data, test_data = Trainer.get_datasets_from_path(config['data_path'],
                                                                       config['scaling_factor'],
                                                                       config['pickle_file_name'])

    input_size = math.prod(test_data.get_sample_shape())  # num processes * num jobs
    hidden_size = config['hidden_size']
    output_size = input_size
    model = MLP(input_size, hidden_size, output_size)

    # move model to device if on gpu
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda:0'
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model.to(device)

    Trainer.checkpoint = checkpoint
    Trainer.load_checkpoint = load_checkpoint

    trainer = Trainer.run(config, model, train_data, valid_data, test_data)
    trainer.train()


def test(config: Config):
    train_data, valid_data, test_data = Trainer.get_datasets_from_path(config['data_path'],
                                                                       config['scaling_factor'],
                                                                       config['pickle_file_name'])

    # build model for test
    input_size = math.prod(test_data.get_sample_shape())  # num processes * num jobs
    hidden_size = config['hidden_size']
    output_size = input_size
    model = MLP(input_size, hidden_size, output_size)

    # move model to device if on gpu
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    trainer = Trainer(config, model, train_data, valid_data, test_data)

    trainer.test()


def run(args):
    tune_config = TUNE_CONFIG

    base_path = args.path
    trainer_config = Config(os.path.abspath('ml/config.yaml'))
    trainer_config['base_path'] = os.path.abspath(base_path)
    trainer_config['data_path'] = os.path.abspath(os.path.join(base_path, 'data'))
    trainer_config['checkpoint_path'] = os.path.abspath(os.path.join(base_path, 'checkpoint'))
    trainer_config['verbose'] = False
    trainer_config['wandb_group'] = 'ray-tune-' + wandb.util.generate_id()

    build_tuner(trainer_config, tune_config, args.samples, args.max_epochs, args.gpus)


if __name__ == '__main__':
    args = parser.parse_args()
    run(args)
