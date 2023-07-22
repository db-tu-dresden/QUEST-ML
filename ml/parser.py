import argparse

parser = argparse.ArgumentParser(description='ML infrastructure for training and testing models')
parser.add_argument('--arch', type=str, help='Model architecture')
parser.add_argument('-g', '--gpu', action='store_true', help='Enable GPU usage')
parser.add_argument('-n', '--n_gpus', type=int, metavar='N', help='How many GPUs to use.If not set, all GPUs are used')
parser.add_argument('--wandb', type=bool, default=True, help='Whether to use wandb is used for logging.')
