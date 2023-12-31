import math

import numpy as np
import torch
import yaml

from ml import Config, Model


class MockArrivalProcess:
    def __init__(self, config, step_size: float, model_direction: int = None, steps: int = None):
        self.config = config
        self.step_size = step_size
        self.model_direction = model_direction
        self.steps = steps

        self.rng = np.random.default_rng(self.config['randomSeed'])

        self.job_arrivals = []
        if config['jobArrivalPath']:
            self.load_job_arrivals()
            self.process_job_arrivals()

        self.current_job_arrivals = []
        self.continue_with_rnd_jobs = self.config['continueWithRndJobs']

        self._now = 0

    def load_job_arrivals(self):
        if self.config['jobArrivalPath']:
            with open(self.config['jobArrivalPath']) as f:
                self.job_arrivals = yaml.full_load(f)

    def process_job_arrivals(self):
        if not self.job_arrivals or len(self.job_arrivals) == 0:
            return

        if self.model_direction == -1:
            max_time = math.ceil(max(elem['time'] for elem in self.job_arrivals))
            for elem in self.job_arrivals:
                elem['time'] = max_time - elem['time']

        if self.steps:
            self.job_arrivals = [elem for elem in self.job_arrivals if elem['time'] <= self.steps]

        self.job_arrivals.sort(key=lambda x: x['time'])

    def step(self) -> torch.Tensor:
        self._now += self.step_size

        if self.job_arrivals is not None:
            self.current_job_arrivals = []

            try:
                while self.job_arrivals[0]['time'] <= self._now:
                    self.current_job_arrivals.append(self.job_arrivals.pop(0))

                types = [job['type'] for job in self.current_job_arrivals]
                job_dist = [sum(1 for _type in types if _type == job_type['name']) for job_type in self.config['jobs']]
                return torch.tensor(job_dist, dtype=torch.float)
            except IndexError:
                if not self.continue_with_rnd_jobs:
                    return torch.zeros(len(self.config['jobs']))

        return torch.tensor([self.rng.poisson(job_type['arrivalProbability'] * self.step_size)
                             for job_type in self.config['jobs']], dtype=torch.float)


class Inferer:
    def __init__(self, ml_config: Config, sys_config, model: Model, target_dist: torch.Tensor,
                 initial_state: torch.Tensor, limit: int, steps: int, k: int, mutate_initial_state: bool,
                 mutation_low: int, mutation_high: int):
        self.ml_config = ml_config
        self.sys_config = sys_config
        self.model = model.eval()
        self.target_dist = target_dist
        self.initial_state = initial_state
        self.limit = limit
        self.steps = steps
        self.k = k
        self.mutate_initial_state = mutate_initial_state
        self.mutation_low = mutation_low
        self.mutation_high = mutation_high

        self.arrival_process = None
        self.model_direction = round(math.tanh(self.ml_config['offset']))
        self.step_size = self.sys_config['loggingRate'] * self.ml_config['scaling_factor']

        if self.model_direction == -1:
            self.initial_state[-1] = self.target_dist

    @staticmethod
    def contains_tgt(state: torch.Tensor, target_dist: torch.Tensor) -> bool:
        return state[0, -1].ge(target_dist).all()

    def step_to_target(self, initial_state: torch.Tensor, target_dist: torch.Tensor, limit: int) -> (int, torch.Tensor):
        state = initial_state
        state = state.unsqueeze(0)      # add batch dim
        step = 0

        with torch.no_grad():
            while not self.contains_tgt(state, target_dist) and step < limit:
                arrivals = self.arrival_process.step()
                state[0, 0] += arrivals
                state = self.model(state)
                step += 1

        return step, state

    def step_through(self, initial_state: torch.Tensor, steps: int) -> (int, torch.Tensor):
        state = initial_state
        state = state.unsqueeze(0)      # add batch dim

        with torch.no_grad():
            for _ in range(steps):
                arrivals = self.arrival_process.step()
                state[0, 0] += arrivals
                state = self.model(state)

        return steps, state

    def predict_target(self, initial_state: torch.Tensor = None) -> (int, torch.Tensor):
        initial_state = initial_state if initial_state is not None else self.initial_state
        target_dist = self.target_dist
        limit = self.limit

        steps, state = self.step_to_target(initial_state.clone(), target_dist, limit)

        if self.contains_tgt(state, target_dist):
            print(f'Target distribution reached after {steps} steps.')
            print(f'Final state is: \n'
                  f'{state.squeeze().numpy()}\n'
                  f'Rounded:\n'
                  f'{state.squeeze().round().numpy()}\n')
        else:
            print(f'Target distribution was NOT contained in the state after {steps} steps.')
            print(f'Final state is: \n'
                  f'{state.squeeze().numpy()}\n'
                  f'Rounded:\n'
                  f'{state.squeeze().round().numpy()}\n')

        return steps, state

    def predict_state(self, initial_state: torch.Tensor = None) -> (int, torch.Tensor):
        initial_state = initial_state if initial_state is not None else self.initial_state
        steps = self.steps

        steps, state = self.step_through(initial_state.clone(), steps)

        print(f'Did {steps} steps from initial state:\n'
              f'{initial_state}\n'
              f'to final state:'
              f'\n{state}')

        return steps, state

    def mutate(self, state: torch.Tensor) -> torch.Tensor:
        shape = state.shape
        mutation = torch.distributions.Uniform(self.mutation_low, self.mutation_high).sample(shape).round()
        return state + mutation

    def run(self, action) -> [torch.Tensor]:
        predictions = []
        initial_state = self.initial_state

        self.arrival_process = MockArrivalProcess(self.sys_config,
                                                  step_size=self.step_size,
                                                  model_direction=self.model_direction,
                                                  steps=self.steps)

        if action == 'STEP_TO_TARGET' and self.model_direction == -1:
            print(f'Warning: Changing action from {action} to STEP_UNTIL, because model direction is negative!')
            action = 'STEP_UNTIL'

        for _ in range(self.k):
            prediction = None

            if action == 'STEP_TO_TARGET':
                prediction = self.predict_target(initial_state=initial_state)
            if action == 'STEP_UNTIL':
                prediction = self.predict_state(initial_state=initial_state)

            predictions.append(prediction)

            if self.mutate_initial_state:
                initial_state = self.mutate(self.initial_state.clone())
                initial_state[initial_state < 0] = 0

        return predictions
