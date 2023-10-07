import argparse
import os.path

import system
from timing import log_runtime

parser = argparse.ArgumentParser(description='ML model for sequence to sequence translation')
parser.add_argument('-p', '--path', help='Path where a config.yaml describing the system and '
                                         'a graph_description.note describing the process graph lie.')
parser.add_argument('-u', '--until', metavar='N', type=int, help='Number of steps until which '
                                                                 'the simulation should run')
parser.add_argument('--plot', default=True, action=argparse.BooleanOptionalAction,
                    help='Enable graph and process distribution plotting')


@log_runtime
def run(args):
    base_path = args.path
    notation_path = os.path.join(base_path, 'graph_description.note')

    config = system.Config(os.path.join(base_path, 'config.yaml'))
    system.simulate(config, notation_path, args.until, plot=args.plot)

    system.simulate(notation_path, save_path, args.plot, args.until)


if __name__ == '__main__':
    args = parser.parse_args()
    run(args)
