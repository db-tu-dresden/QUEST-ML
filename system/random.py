import numpy as np


class RandomContainer:
    def __init__(self, rng: np.random.Generator, mean: float = None, std: float = None, beta: float = None):
        self.rng = rng
        self.mean = mean
        self.std = std
        self.beta = beta
