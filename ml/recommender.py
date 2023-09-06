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
                 initial_state: torch.Tensor, limit: int):
        self.ml_config = ml_config
        self.sys_config = sys_config
        self.model = model
        self.target_dist = target_dist
        self.initial_state = initial_state
        self.limit = limit

        self.arrival_process = MockArrivalProcess(self.sys_config)

    def contains_tgt(self, state: torch.tensor):
        return state[0, -1].ge(self.target_dist).all()

    def predict_forward(self):
        state = self.initial_state
        state = state.unsqueeze(0)      # add batch dim
        step = 0

        while not self.contains_tgt(state) and step < self.limit:
            state[0, 0] += self.arrival_process.step()
            state = self.model(state)
            step += 1

        state = state if self.contains_tgt(state) else None

        return step, state

    def predict(self):
        step, state = self.predict_forward()

        if state is not None:
            print(f'Target distribution reached after {step} steps.')
            print(f'Target distribution is: {state[-1]}')
        else:
            print(f'Target distribution was NOT contained in the state after {step} steps.')
