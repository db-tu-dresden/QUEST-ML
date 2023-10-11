import argparse
import os

import torch
import yaml
from torcheval.metrics.functional import mean_squared_error

import system
from ml import Config as MLConfig, Parser
from ml.config import InferenceConfig
from ml.inferer import Inferer
from ml.models import build_model
from notation import Notation
from system import Config as SysConfig


def add_subparsers(parser):
    subparsers = parser.add_subparsers(help='Actions', parser_class=argparse.ArgumentParser, dest='action')

    step_to_parser = subparsers.add_parser('STEP_TO_TARGET')
    step_to_parser.add_argument('--limit', type=int, metavar='N', default=10,
                                help='Maximal number of steps to be tested')

    step_through_parser = subparsers.add_parser('STEP_UNTIL')
    step_through_parser.add_argument('--steps', type=int, metavar='N', required=True,
                                     help='Number of steps to take from initial state')


def run_simulation(args: argparse.Namespace, sys_config: SysConfig, ml_config: MLConfig,
                   initial_state: torch.Tensor, tgt_dist: torch.Tensor):
    print(f'Running simulation {args.k_simulation} times...')
    notation_path = os.path.join(ml_config['base_path'], 'graph_description.note')
    with open(notation_path) as f:
        notation_string = f.read()
    notation = Notation.parse(notation_string)

    _initial_state = initial_state.round().int().numpy()

    simulation_data = []

    if args.action == 'STEP_TO_TARGET':
        simulation_data = system.simulate_to_target(sys_config, notation, _initial_state, tgt_dist,
                                                    k=args.k_simulation, verbose=args.verbose)
    if args.action == 'STEP_UNTIL':
        simulation_data = system.simulate_from_state(sys_config, notation, _initial_state, args.steps,
                                                     k=args.k_simulation, verbose=args.verbose)

    quickest = sorted(simulation_data, key=lambda x: x['steps'])[0]

    _job_arrival_path = os.path.join(ml_config['base_path'], '_job_arrivals.yaml')
    with open(_job_arrival_path, 'w') as f:
        yaml.dump(quickest['job_arrivals'], f)

    sys_config['jobArrivalPath'] = _job_arrival_path
    sys_config['continueWithRndJobs'] = True

    return simulation_data


def find_closest(predictions: [torch.Tensor], simulations: [torch.Tensor]):
    mean = torch.stack(simulations).mean(axis=0, dtype=torch.float)
    mses = torch.stack([mean_squared_error(mean, pred.squeeze()) for pred in predictions])

    min_index = mses.argmin()

    print(f'\nPrediction with lowest MSE to simulation mean is:\n'
          f'{predictions[min_index].numpy()}\n'
          f'Rounded:\n'
          f'{predictions[min_index].round().numpy()}\n\n'
          f'MSE is {mses[min_index]}\n')

    found_identical = False
    for pred in predictions:
        pred = pred.squeeze()

        for sim in simulations:
            if pred.equal(sim):
                print(f'Prediction {pred} is equal to simulation result {sim}!')
                found_identical = True

    if not found_identical:
        print('No identical simulations to predictions found!')


def run():
    cwd = os.path.dirname(os.path.realpath(__file__))
    ml_config = MLConfig(os.path.join(cwd, 'ml/config.yaml'))

    parser = Parser(ml_config)
    parser.add_argument('--k_model', metavar='N', type=int, default=1,
                        help='K for k-fold validation of the model')
    parser.add_argument('--k_simulation', metavar='N', type=int, default=1,
                        help='K for k-fold validation of the simulation')
    parser.add_argument('--mutate', action=argparse.BooleanOptionalAction,
                        help='If set mutate the initial state between different runs. '
                             'Useful in combination with k-fold validation.'
                             'The first run is always without mutation.'
                             'Mutation is done by adding a tensor rounded to integers, '
                             'sampled from a uniform distribution.'
                             'Negative values in the resulting state are set to zero.'
                             'Default: Is true if argument k_model > 1.')
    parser.add_argument('--mutation_low', metavar='N', type=int, default=-2,
                        help='Low value of the uniform probability distribution used for mutation')
    parser.add_argument('--mutation_high', metavar='N', type=int, default=2,
                        help='Low value of the uniform probability distribution used for mutation')
    parser.add_argument('--verbose', '-v', default=True, action=argparse.BooleanOptionalAction,
                        help='If set prints simulation output')
    parser.add_argument('--job_arrival_path', type=str, help='Path to yaml file containing job arrivals')

    args = parser.parse_args(post_arch_arg_add_fn=add_subparsers)

    ml_config.update_from_args(args)
    ml_config['load_model'] = True

    model = build_model(ml_config)
    model.load(ml_config)

    sys_config = SysConfig(os.path.join(ml_config['base_path'], 'config.yaml'))
    sys_config['jobArrivalPath'] = args.job_arrival_path or sys_config['jobArrivalPath']

    inf_config = InferenceConfig(os.path.join(ml_config['base_path'], 'inference_config.yaml'))
    initial_state = torch.tensor(inf_config['initialState'], dtype=torch.float, requires_grad=False)
    tgt_dist = torch.tensor(inf_config['targetDist'], dtype=torch.float, requires_grad=False)

    # +1 for the implicit exit process
    assert initial_state.shape == (len(sys_config['processes']) + 1, len(sys_config['jobs']))

    if args.action == 'STEP_TO_TARGET':
        assert tgt_dist.shape == (len(sys_config['jobs']),)

    ###################
    # Run simulations #
    ###################

    simulation_data = run_simulation(args, sys_config, ml_config, initial_state, tgt_dist)

    #################
    # Run inference #
    #################

    inferer = Inferer(ml_config, sys_config, model,
                      target_dist=tgt_dist,
                      initial_state=initial_state,
                      limit=getattr(args, 'limit', None),
                      steps=getattr(args, 'steps', round(max(sim['steps'] for sim in simulation_data))),
                      k=args.k_model,
                      mutate_initial_state=args.mutate if args.mutate is not None else args.k_model > 1,
                      mutation_low=args.mutation_low,
                      mutation_high=args.mutation_high)
    predictions = inferer.run(action=args.action)

    ##############################
    # Compare / Evaluate Results #
    ##############################

    print(f'Finding closest prediction...')
    find_closest([state for _, state in predictions], [torch.tensor(elem['final_state']) for elem in simulation_data])


if __name__ == '__main__':
    run()
