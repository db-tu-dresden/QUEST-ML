import numpy as np
import simpy

from config import Config
from notation import Notation


class Job:
    def __init__(self, job_id: int, job_type: str, failure_rate: float):
        self.id = job_id
        self.job_type = job_type
        self.failure_rate = failure_rate

        self.log = {}

    def __str__(self) -> str:
        return f'{self.id}({self.job_type})'


class Queue(simpy.Store):
    def __init__(self, env, capacity=1, name='default', discipline='FIFO'):
        super().__init__(env, capacity)
        self.name = name
        self.log = {}
        self.state = {}
        self.discipline = discipline  # is default of store object


class System:
    def __init__(self, config: Config, notation: Notation):
        if config is None or not isinstance(config, Config):
            raise ValueError(f'Config must be specified and of type Config. Got {repr(config)}')
        if notation is None or not isinstance(notation, Notation):
            raise ValueError(f'Notation must be specified and of type Notation. Got {repr(notation)}')
        self.config = config
        self.notation = notation

        self.random = np.random.default_rng(seed=42)

        self.env = simpy.Environment()
        self.break_event = self.env.event()
        self.queues = []
        self.processes = {}

    def add_queue(self, name: str = None):
        self.queues.append(Queue(self.env, capacity=np.inf, name=name))

    def build_queues(self):
        if self.notation.seq is None:
            return

        self.queues = []
        elem = self.notation.seq.next

        i = 1
        while elem:
            self.add_queue(name=f'Queueing System {i}')
            i += 1
        self.add_queue(name='Out')

    def get_rand_job(self, job_id):
        idx = self.random.choice(range(len(self.config['jobList'])), p=self.config['jobDistribution'])
        job_type = self.config['jobList'][idx]
        failure_rate = self.config['failure_rate'][idx]
        return Job(job_id, job_type, failure_rate)

    def add_rand_job(self, job_id):
        if not len(self.queues):
            raise Exception('Can not add a job to empty list of queues.')

        job = self.get_rand_job(job_id)
        t = self.get_arrival_timeout()
        yield self.env.timeout(t)
        yield self.queues[0].put(job)

    def get_arrival_timeout(self):
        return self.random.exponential()

    def add_arrival_process(self):
        job_id = 0
        while hasattr(self.config.data, 'until') or job_id < self.config['jobLimit']:
            job_id += 1
            self.add_rand_job(job_id)

    def add_process(self):
        pass

    def build(self):
        if self.notation is None or self.notation.seq is None:
            return

        self.build_queues()
        self.add_arrival_process()

    def simulate(self):
        pass


def main():
    path = 'graph_description.note'
    with open(path) as f:
        text = f.read()

    notation = Notation.parse(text)

    config = Config('config.yaml')

    system = System(config, notation)
    system.build()


if __name__ == '__main__':
    main()
