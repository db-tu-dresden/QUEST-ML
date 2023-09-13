import argparse
import os
from datetime import datetime

import numpy as np
import torch

from ml import Config as MLConfig, Parser
from ml.models import build_model
from ml.recommender import Recommender
from notation import Notation
from system import Config as SysConfig, Environment, System


def add_subparsers(parser):
    subparsers = parser.add_subparsers(help='Actions', parser_class=argparse.ArgumentParser, dest='action')

    step_to_parser = subparsers.add_parser('STEP_TO')
    step_to_parser.add_argument('--tgt', nargs='+', type=int, required=True,
                                help='The target distribution')
    step_to_parser.add_argument('--limit', type=int, metavar='N', default=10,
                                help='Maximal number of steps to be tested')

    step_through_parser = subparsers.add_parser('STEP_THROUGH')
    step_through_parser.add_argument('--steps', type=int, metavar='N', required=True,
                                     help='Number of steps to take from initial state')


def simulate_from_state(state, steps, sys_config, ml_config, k=1, verbose=False):
    with open(os.path.join(ml_config['base_path'], 'graph_description.note')) as f:
        text = f.read()
    notation = Notation.parse(text)

    final_states = []
    for i in range(k):
        env = Environment()
        sys_config['randomSeed'] = int(''.join(str(el) for el in datetime.now().timestamp().as_integer_ratio()))
        system = System(sys_config, notation, env=env)
        system.set_state(state)
        system.run(steps)

        final_state = system.logger.get_current_state()
        final_states.append(final_state)
        if verbose:
            print(f'Final state in run {i + 1} of {k} is:\n'
                  f'{final_state}\n')

    final_states = np.stack(final_states)
    mean = final_states.mean(axis=0)
    std = final_states.std(axis=0)
    print(f'Mean is: \n'
          f'{mean}\n')
    print(f'Standard Deviation is: \n'
          f'{std}\n')


def run():
    cwd = os.path.dirname(os.path.realpath(__file__))
    ml_config = MLConfig(os.path.join(cwd, 'ml/config.yaml'))

    parser = Parser(ml_config)
    parser.add_argument('--k', '-k', metavar='N', type=int, default=1, help='K for k-fold validation')
    parser.add_argument('--verbose', '-v', default=False, action=argparse.BooleanOptionalAction,
                        help='If set individual final states of the simulation are printed')
    parser.add_argument('--job_arrival_path', type=str, help='Path to yaml file containing job arrivals')

    args = parser.parse_args(post_arch_arg_add_fn=add_subparsers)

    ml_config.update_from_args(args)
    ml_config['load_model'] = True

    model = build_model(ml_config)
    model.load(ml_config)

    sys_config = SysConfig(os.path.join(ml_config['base_path'], 'config.yaml'))
    sys_config['jobArrivalPath'] = args.job_arrival_path or sys_config['jobArrivalPath']

    initial_state = torch.zeros(len(sys_config['processes']) + 1, len(sys_config['jobs']),
                                requires_grad=False)

    recommender = Recommender(ml_config, sys_config, model,
                              target_dist=torch.tensor(args.tgt) if hasattr(args, 'tgt') else None,
                              initial_state=initial_state,
                              limit=args.limit if hasattr(args, 'limit') else None,
                              steps=args.steps if hasattr(args, 'steps') else None)
    steps, state = recommender.run(action=args.action)

    state = state.squeeze().round().int().numpy()
    initial_state = initial_state.round().int().numpy()

    simulate_from_state(initial_state, steps, sys_config, ml_config, args.k, args.verbose)


if __name__ == '__main__':
    run()
