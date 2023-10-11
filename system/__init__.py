import os
from datetime import datetime

import numpy as np
import torch

from notation import Notation
from system.config import Config
from system.environment import Environment
from system.job import Job
from system.logger import Logger
from system.process import Process, ArrivalProcess, ExitProcess
from system.queue import Queue
from system.system import System


def print_run_stats(run_data: list, k: int):
    print(f'Ran simulation {k} times')
    print(f'Steps: {np.mean([elem["steps"] for elem in run_data])} '
          f'± {np.std([elem["steps"] for elem in run_data])}')
    print(f'Runtime: {np.mean([elem["runtime"] for elem in run_data])} sec '
          f'± {np.std([elem["runtime"] for elem in run_data])} sec')
    print(f'Final states: \n'
          f'Mean\n'
          f'{np.mean([elem["final_state"] for elem in run_data], axis=0)}\n'
          f'Std\n'
          f'{np.std([elem["final_state"] for elem in run_data], axis=0)}\n')


def simulate(config: Config, notation: Notation, steps: int, k: int = 1, initial_state: torch.Tensor = None,
             verbose=False, vary_random_seed: bool = True, plot: bool = False):
    if plot:
        notation.draw(os.path.join(config['base_path'], 'graph.png'))

    run_data = []

    for i in range(k):
        env = Environment()
        if vary_random_seed:
            config['randomSeed'] = int(''.join(str(el) for el in datetime.now().timestamp().as_integer_ratio()))

        sys = System(config, notation, env=env)
        if initial_state is not None:
            sys.set_state(initial_state)
        sys.run(steps)

        if plot:
            suffix = f'_{k}' if k > 1 else ''
            sys.logger.plot(path=os.path.join(config['base_path'], f'dist{suffix}.png'))

        run_data.append({
            'steps': sys.logger.get_steps(),
            'runtime': sys.logger.get_runtime(),
            'final_state': sys.logger.get_state(),
            'job_arrivals': sys.logger.get_job_arrivals()
        })

    if verbose:
        print_run_stats(run_data, k)

    return run_data


def simulate_from_state(config: Config, notation: Notation, initial_state: torch.Tensor, steps: int, k: int = 1,
                        verbose=False, vary_random_seed: bool = True, plot: bool = False):
    return simulate(config, notation, steps, k, initial_state, verbose, vary_random_seed, plot)


def simulate_to_target(config: Config, notation: Notation, initial_state: torch.Tensor, target_dist: torch.Tensor,
                       k: int = 1, verbose: bool = False, vary_random_seed: bool = True, plot: bool = False):
    def break_on_target(process: ExitProcess):
        return all(target_dist[i] <= job_count for i, (_, job_count) in enumerate(process.job_dist.items()))

    if plot:
        notation.draw(os.path.join(config['base_path'], 'graph.png'))

    run_data = []

    for i in range(k):
        env = Environment()
        if vary_random_seed:
            config['randomSeed'] = int(''.join(str(el) for el in datetime.now().timestamp().as_integer_ratio()))

        sys = System(config, notation, env=env)
        sys.set_break_condition(break_on_target)
        sys.set_state(initial_state)
        sys.run()

        if plot:
            suffix = f'_{k}' if k > 1 else ''
            sys.logger.plot(path=os.path.join(config['base_path'], f'dist{suffix}.png'))

        run_data.append({
            'steps': sys.logger.get_steps(),
            'runtime': sys.logger.get_runtime(),
            'final_state': sys.logger.get_state(),
            'job_arrivals': sys.logger.get_job_arrivals()
        })

    if verbose:
        print_run_stats(run_data, k)

    return run_data
