import numpy as np
import simpy


class Environment(simpy.Environment):
    def __init__(self):
        super().__init__()
        self.rng = np.random.default_rng(seed=42)
