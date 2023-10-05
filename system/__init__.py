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


def simulate_from_state(config: Config, notation_path: str, state: torch.Tensor, steps: int, k: int = 1, verbose=False):
    with open(notation_path) as f:
        text = f.read()
    notation = Notation.parse(text)

    final_states = []
    for i in range(k):
        env = Environment()
        config['randomSeed'] = int(''.join(str(el) for el in datetime.now().timestamp().as_integer_ratio()))
        sys = System(config, notation, env=env)
        sys.set_state(state)
        sys.run(steps)

        final_state = sys.logger.get_state()
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
          f'{std}\n\n')

    return list(zip([steps] * k, [torch.tensor(state) for state in final_states]))


def simulate_to_target(config: Config, notation_path: str, initial_state: torch.Tensor, target_dist: torch.Tensor):
    def break_on_target(process: ExitProcess):
        return all(target_dist[i] <= job_count for i, (_, job_count) in enumerate(process.job_dist.items()))

    with open(notation_path) as f:
        notation_string = f.read()

    notation = Notation.parse(notation_string)

    env = Environment()

    sys = System(config, notation, env=env)
    sys.set_break_condition(break_on_target)
    sys.set_state(initial_state)
    sys.run()
