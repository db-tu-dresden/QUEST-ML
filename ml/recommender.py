import torch

from ml import Config, Model


class Recommender:
    def __init__(self, ml_config: Config, model: Model, target_dist: torch.Tensor, initial_state: torch.Tensor):
        self.ml_config = ml_config
        self.model = model
        self.target_dist = target_dist
        self.initial_state = initial_state

    def predict_forward(self):
        state = self.initial_state
        state[0] += self.target_dist
        state = state.unsqueeze(0)      # add batch dim
        step = 0

        while not state[0, -1].ge(self.target_dist).all():
            state = self.model(state)
            step += 1

        return step, state

    def predict(self):
        step, state = self.predict_forward()

        print(f'Target distribution reached after {step} steps.')
        print(f'Target distribution is: {state[-1]}')
