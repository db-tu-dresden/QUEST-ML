from __future__ import annotations

from collections import defaultdict

import numpy as np
import simpy

from config import Config
from notation import Notation


class Environment(simpy.Environment):
    def __init__(self):
        super().__init__()
        self.rng = np.random.default_rng(seed=42)


class JobType:
    def __init__(self, name: str, arrival_prob: float, failure_rate: float, env: Environment):
        self.name = name
        self.arrival_prob = arrival_prob
        self.failure_rate = failure_rate
        self.env = env

    def __eq__(self, other):
        return hasattr(other, 'name') and self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return f'JobType(name={self.name}, arrival_prob={self.arrival_prob}, failure_rate={self.failure_rate})'


class JobTypeCollection:

    def __init__(self, types: {JobType}, env: Environment):
        super().__init__()
        self.types = types
        self.env = env

    def get_rand_job(self, job_id):
        job_type = self.env.rng.choice(list(self.types), p=[job_type.arrival_prob for job_type in self.types])
        return Job(job_id, job_type, env=self.env)

    @classmethod
    def from_config(cls, config: Config, env: Environment):
        types = set(JobType(job['name'], job['arrivalProbability'], job['failureRate'], env=env)
                    for job in config['jobs'])
        return cls(types, env=env)

    def __repr__(self):
        return f'JobTypeCollection(types={repr(set(type for type in self.types))})'


class Job:
    def __init__(self, id: int, type: JobType, env: Environment):
        super().__init__()
        self.id = id
        self.type = type
        self.env = env

    def service(self):
        t = self.env.rng.normal(loc=1.0, scale=0.2)
        yield self.env.timeout(t)

    def __repr__(self) -> str:
        return f'Job(id={self.id}, type={repr(self.type)})'


class Queue(simpy.Store):
    def __init__(self, data: {str}, env: Environment, capacity=np.inf, name='default'):
        super().__init__(env, capacity)
        self.name = name
        self.data = data


class Process:
    def __init__(self, env: Environment, queue: Queue = None):
        super().__init__()
        self.queue = queue
        self.next = {}

        self.env = env

    def update_next(self, update_dict: {str: Process}):
        self.next.update(update_dict)

    def push(self, job: Job):
        yield self.queue.put(job)

    def process(self):
        job = yield self.queue.get()

        yield self.env.process(job.service())

        yield self.env.process(self.next[job.type.name].push(job))

    def run(self):
        while True:
            yield from self.process()


class ArrivalProcess(Process):
    def __init__(self, job_types: JobTypeCollection, env: Environment):
        super().__init__(env)
        self.job_types = job_types
        self.last_job_id = -1

    def process(self):
        t = self.env.rng.exponential()
        yield self.env.timeout(t)

        self.last_job_id += 1
        job = self.job_types.get_rand_job(self.last_job_id)

        yield self.env.process(self.next[job.type.name].push(job))

    def run(self):
        while True:
            yield from self.process()


class ExitProcess(Process):
    def process(self):
        yield self.env.timeout(1)

    def run(self):
        while True:
            yield from self.process()


class System:
    def __init__(self, config: Config, notation: Notation, env: Environment):
        super().__init__()
        if config is None or not isinstance(config, Config):
            raise ValueError(f'Config must be specified and of type Config. Got {repr(config)}')
        if notation is None or not isinstance(notation, Notation):
            raise ValueError(f'Notation must be specified and of type Notation. Got {repr(notation)}')

        self.config = config
        self.notation = notation
        self.data = self.notation.data.value

        self.env = env

        self.job_types = JobTypeCollection.from_config(self.config, env=self.env)
        self.processes = {}

    def get_job_types(self):
        return [JobType(job['name'], job['arrivalProbability'], job['failureRate'], env=self.env)
                for job in self.config['jobs']]

    def build_processes(self):
        if not self.notation.graph:
            raise Exception('Can not build system with no graph specified.')

        nodes = reversed(list(self.notation.graph.nodes(data=True)))

        for node, props in nodes:
            queue = Queue(props['data'], env=self.env)
            process = Process(queue=queue, env=self.env)
            self.processes[node] = process

            for _, out, props in self.notation.graph.edges(node, data=True):
                process.update_next({datum: self.processes[out] for datum in props['data']})

    def add_arrival_and_exit_process(self):
        arrival_process = ArrivalProcess(self.job_types, env=self.env)
        self.processes[-1] = arrival_process

        queue = Queue(self.data, env=self.env)
        exit_process = ExitProcess(queue=queue, env=self.env)
        last_process_id, last_process = sorted(self.processes.items(), reverse=True)[0]
        self.processes[last_process_id + 1] = exit_process

        arrival_process.next = defaultdict(lambda: self.processes[0])
        last_process.next = defaultdict(lambda: exit_process)

    def build(self):
        if not self.notation:
            raise Exception('Can not build system with no notation specified.')

        self.build_processes()
        self.add_arrival_and_exit_process()

    def run(self):
        for _, process in sorted(self.processes.items()):
            self.env.process(process.run())

        self.env.run(until=100)


def main():
    path = 'graph_description.note'
    with open(path) as f:
        text = f.read()

    notation = Notation.parse(text)

    config = Config('config.yaml')

    env = Environment()

    system = System(config, notation, env=env)
    system.build()
    system.run()


if __name__ == '__main__':
    main()
