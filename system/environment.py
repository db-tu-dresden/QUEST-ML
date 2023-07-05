from typing import Optional, Any

import simpy
from simpy import Timeout
from simpy.core import SimTime


class Environment(simpy.Environment):
    def __init__(self):
        super().__init__()
        self.system = None

    def timeout(self, delay: SimTime = 0, value: Optional[Any] = None) -> Timeout:
        if delay < 0:
            delay = 0
        return super().timeout(delay, value)
