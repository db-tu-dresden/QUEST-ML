import os

from ml import seed, Config, Parser, Trainer
from ml.models import build_model


def run():
    cwd = os.path.dirname(os.path.realpath(__file__))
    config = Config(os.path.join(cwd, '../config.yaml'))

    seed(config['seed'])

    parser = Parser(config)
    args = parser.parse_args()

    config.update_from_args(args)

    model = build_model(config)

    train_data, valid_data, test_data = Trainer.get_datasets(config)
    trainer = Trainer(config, model, train_data, valid_data, test_data)
    trainer.test()
