from ml import Config, Trainer, Parser
from ml.models import build_model


def run():
    config = Config('ml/config.yaml')

    parser = Parser(config)
    args = parser.parse_args()

    config.update_from_args(args)

    model = build_model(config)

    Trainer.run(config, model)


if __name__ == '__main__':
    run()
