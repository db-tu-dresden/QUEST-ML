import simpy


class Environment(simpy.Environment):
    def __init__(self):
        super().__init__()
        self.system = None
