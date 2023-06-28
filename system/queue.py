import numpy as np
import simpy

from system.environment import Environment


class Queue(simpy.Store):
    def __init__(self, data: {str}, env: Environment, capacity=np.inf, name='default'):
        super().__init__(env, capacity)
        self.name = name
        self.data = data

    def __repr__(self):
        cls = self.__class__.__name__
        return f'{cls}(data={self.data!r}, env={self._env!r}, capacity={self._capacity!r}, name={self.name!r})'
