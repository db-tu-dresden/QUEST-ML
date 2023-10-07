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


def simulate(notation_path: str, save_path: str, plot: bool, until: int):
    with open(notation_path) as f:
        notation_string = f.read()

    notation = Notation.parse(notation_string)
    if plot:
        notation.draw(os.path.join(save_path, 'graph.png'))

    config = Config(os.path.join(save_path, 'config.yaml'))

    env = Environment()

    sys = System(config, notation, env=env)
    sys.run(until)
    if plot:
        sys.logger.plot(path=os.path.join(save_path, 'dist.png'))


def simulate_from_state(config: Config, notation_path: str, initial_state: torch.Tensor, steps: int, k: int = 1,
                        verbose=False, vary_random_seed: bool = True):
    with open(notation_path) as f:
        text = f.read()
    notation = Notation.parse(text)

    run_data = []

    for i in range(k):
        env = Environment()
        if vary_random_seed:
            config['randomSeed'] = int(''.join(str(el) for el in datetime.now().timestamp().as_integer_ratio()))

        sys = System(config, notation, env=env)
        sys.set_state(initial_state)
        sys.run(steps)

        run_data.append({
            'steps': sys.logger.get_steps(),
            'runtime': sys.logger.get_runtime(),
            'final_state': sys.logger.get_state(),
            'job_arrivals': sys.logger.get_job_arrivals()
        })

    if verbose:
        print_run_stats(run_data, k)

    return run_data


def simulate_to_target(config: Config, notation_path: str, initial_state: torch.Tensor, target_dist: torch.Tensor,
                       k: int = 1, verbose: bool = False, vary_random_seed: bool = True):
    def break_on_target(process: ExitProcess):
        return all(target_dist[i] <= job_count for i, (_, job_count) in enumerate(process.job_dist.items()))

    with open(notation_path) as f:
        notation_string = f.read()

    notation = Notation.parse(notation_string)

    run_data = []

    for i in range(k):
        env = Environment()
        if vary_random_seed:
            config['randomSeed'] = int(''.join(str(el) for el in datetime.now().timestamp().as_integer_ratio()))

        sys = System(config, notation, env=env)
        sys.set_break_condition(break_on_target)
        sys.set_state(initial_state)
        sys.run()

        run_data.append({
            'steps': sys.logger.get_steps(),
            'runtime': sys.logger.get_runtime(),
            'final_state': sys.logger.get_state(),
            'job_arrivals': sys.logger.get_job_arrivals()
        })

    if verbose:
        print_run_stats(run_data, k)

    return run_data
