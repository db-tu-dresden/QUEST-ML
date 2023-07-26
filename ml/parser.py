import argparse
import sys

from ml import Config
from ml.models import parse_arch


class Parser(argparse.ArgumentParser):

    def __init__(self, config: Config, description: str = 'ML infrastructure for training and testing models',
                 *args, **kwargs):
        super().__init__(*args, description=description, conflict_handler='resolve', **kwargs)
        self.config = config
        self.add_model_args()

        for name, v in self.config.config_dict.items():
            if v['type'] == bool:
                self.add_argument(f'--{name}',
                                  default=self.config[name],
                                  action=argparse.BooleanOptionalAction)
                continue

            self.add_argument(f'--{name}',
                              type=v['type'],
                              metavar='N' if v['type'] == int else None,
                              default=self.config[name])

        self.add_argument('-p', '--path', dest='base_path', required=True,
                          help='Path where a config.yaml describing the system and '
                               'a graph_description.note describing the process graph lie.')

    def add_model_args(self):
        group = self.add_argument_group('Model configuration')

        from ml.models import ARCH_MODEL_REGISTRY
        group.add_argument('--arch', '-a', metavar='ARCH',
                           choices=ARCH_MODEL_REGISTRY.keys(),
                           required=True,
                           help='model architecture')

    def format_help(self):
        msg = super().format_help()
        args = sys.argv[1:]

        ident = None

        if '--arch' in args:
            ident = '--arch'
        elif '-a' in args:
            ident = '-a'

        if ident is None:
            return msg

        import ml
        arch = args[args.index(ident) + 1]
        parser_id = arch + '_parser'

        if parser_id not in vars(ml.models):
            return msg

        return vars(ml.models)[parser_id].format_help()

    def parse_args(self, *args, **kwargs):
        return parse_arch(super(Parser, self))
