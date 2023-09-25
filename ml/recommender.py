import numpy as np
import torch

from ml import Config, Model


class MockArrivalProcess:
    def __init__(self, config):
        self.config = config
        self.rng = np.random.default_rng(self.config['randomSeed'])

    def step(self):
        return torch.tensor([self.rng.poisson(job_type['arrivalProbability']) for job_type in self.config['jobs']])


class Recommender:
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

        self.arrival_process = MockArrivalProcess(self.sys_config)

    @staticmethod
    def contains_tgt(state: torch.Tensor, target_dist: torch.Tensor):
        return state[0, -1].ge(target_dist).all()

    def step_to_target(self, initial_state: torch.Tensor, target_dist: torch.Tensor, limit: int):
        state = initial_state
        state = state.unsqueeze(0)      # add batch dim
        step = 0

        with torch.no_grad():
            while not self.contains_tgt(state, target_dist) and step < limit:
                state[0, 0] += self.arrival_process.step()
                state = self.model(state)
                step += 1

        state = state if self.contains_tgt(state, target_dist) else None

        return step, state

    def step_through(self, initial_state: torch.Tensor, steps: int):
        state = initial_state
        state = state.unsqueeze(0)      # add batch dim

        with torch.no_grad():
            for _ in range(steps):
                state = self.model(state)

        return steps, state

    def predict_target(self, initial_state: torch.Tensor = None, target_dist: torch.Tensor = None, limit: int = None):
        initial_state = initial_state if initial_state is not None else self.initial_state
        target_dist = target_dist if target_dist is not None else self.target_dist
        limit = limit if limit is not None else self.limit

        steps, state = self.step_to_target(initial_state, target_dist, limit)

        if state is not None:
            print(f'Target distribution reached after {steps} steps.')
            print(f'Target distribution is: {state[-1]}')
        else:
            print(f'Target distribution was NOT contained in the state after {steps} steps.')

        return steps, state

    def predict_state(self, initial_state: torch.Tensor = None, steps: int = None):
        initial_state = initial_state if initial_state is not None else self.initial_state
        steps = steps if steps is not None else self.steps

        steps, state = self.step_through(initial_state, steps)

        print(f'Did {steps} steps from initial state:\n'
              f'{self.initial_state}\n'
              f'to final state:'
              f'\n{state}')

        return steps, state

    def mutate(self, state: torch.Tensor):
        shape = state.shape
        mutation = torch.distributions.Uniform(self.mutation_low, self.mutation_high).sample(shape).round()
        return state + mutation

    def run(self, action):
        predictions = []
        initial_state = self.initial_state

        if action == 'STEP_TO':
            for _ in range(self.k):
                predictions.append(self.predict_target(initial_state=initial_state))
                if self.mutate_initial_state:
                    initial_state = self.mutate(self.initial_state)
                    initial_state[initial_state < 0] = 0

        if action == 'STEP_THROUGH':
            for _ in range(self.k):
                predictions.append(self.predict_state(initial_state=initial_state))
                if self.mutate_initial_state:
                    initial_state = self.mutate(self.initial_state)
                    initial_state[initial_state < 0] = 0

        return predictions
