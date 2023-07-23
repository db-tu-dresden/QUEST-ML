import argparse
import sys


def add_model_args(parser):
    group = parser.add_argument_group('Model configuration')

    from ml.models import ARCH_MODEL_REGISTRY
    group.add_argument('--arch', '-a', metavar='ARCH',
                       choices=ARCH_MODEL_REGISTRY.keys(),
                       required=True,
                       help='model architecture')
    return group


def override_format_help(parser):
    old_format_help = parser.format_help

    def format_help():
        msg = old_format_help()
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

    parser.format_help = format_help


parser = argparse.ArgumentParser(description='ML infrastructure for training and testing models')
parser.add_argument('-g', '--gpu', action='store_true', help='Enable GPU usage')
parser.add_argument('-n', '--n_gpus', type=int, metavar='N', help='How many GPUs to use.If not set, all GPUs are used')
parser.add_argument('--wandb', type=bool, default=True, help='Whether to use wandb is used for logging.')

add_model_args(parser)
override_format_help(parser)
