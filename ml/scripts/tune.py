import os.path

import torch
import wandb
from ray import tune
from ray.air import session, RunConfig
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler
from torch import nn

from ml import Trainer, Config, Parser, seed
from ml.models import build_model
from ml.scripts.tune_config import TUNE_CONFIG


def build_tuner(config: Config, tune_config: dict, num_samples: int = 10, max_num_epochs: int = 10,
                gpus_per_trial: float = 2):
    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train, config=config),
            resources={'cpu': 1, 'gpu': gpus_per_trial}
        ),
        tune_config=tune.TuneConfig(
            metric='loss',
            mode='min',
            scheduler=scheduler,
            num_samples=num_samples,
            reuse_actors=False,
        ),
        run_config=RunConfig(verbose=int(config['verbose'])),
        param_space=tune_config,
    )

    results = tuner.fit()

    best_result = results.get_best_result('loss', 'min')

    print(f'Best trial config: {best_result.config}')
    print(f'Best trial final validation loss: {best_result.metrics["loss"]}')
    print(f'Best trial final validation accuracy: {best_result.metrics["accuracy"]}')

    config.update(best_result.config)

    test(config)


def save_checkpoint(trainer: Trainer, valid_loss: float, valid_accuracy: float):
    os.makedirs(trainer.config['save_dir'], exist_ok=True)
    torch.save((
        trainer.model.state_dict(),
        trainer.optimizer.state_dict()
    ), trainer.config['checkpoint_save_path'])

    checkpoint = Checkpoint.from_directory(trainer.config['save_dir'])
    session.report({'loss': valid_loss, 'accuracy': valid_accuracy}, checkpoint=checkpoint)


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

    model = build_model(config)

    # move model to device if on gpu
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda:0'
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model.to(device)
    config['device'] = device

    Trainer.save_checkpoint = save_checkpoint
    Trainer.load_checkpoint = load_checkpoint

    Trainer.run(config, model)


def test(config: Config):
    # build model for test
    model = build_model(config)

    train_data, valid_data, test_data = Trainer.get_datasets(config)

    # move model to device if on gpu
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    config['device'] = device

    trainer = Trainer(config, model, train_data, valid_data, test_data)

    trainer.test()

    model.save(config)


def run():
    tune_config = TUNE_CONFIG

    cwd = os.path.dirname(os.path.realpath(__file__))
    trainer_config = Config(os.path.join(cwd, '../config.yaml'))

    seed(trainer_config['seed'])

    parser = Parser(trainer_config, 'Raytune for the task')
    parser.add_argument('-s', '--samples', type=int,
                        help='Number of times to sample from the hyperparameter space.', default=10)
    parser.add_argument('-e', '--max-epochs', type=int, help='Max number of epochs per trail.', default=10)
    parser.add_argument('-g', '--gpus', type=float, help='GPUs used per trail.', default=0)
    args = parser.parse_args()

    args.base_path = os.path.abspath(args.base_path)
    args.save_dir = args.save_dir or os.path.join(args.base_path, 'raytune')
    trainer_config.update_from_args(args)

    trainer_config['verbose'] = False
    trainer_config['wandb_group'] = 'ray-tune-' + wandb.util.generate_id()

    build_tuner(trainer_config, tune_config, args.samples, args.max_epochs, args.gpus)
