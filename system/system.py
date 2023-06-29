from __future__ import annotations

from collections import defaultdict

import numpy as np

from notation import Notation
from system.config import Config
from system.environment import Environment
from system.job import JobTypeCollection
from system.logger import Logger
from system.process import Process, ArrivalProcess, ExitProcess
from system.queue import Queue
from system.random import RandomContainer


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
        env.system = self

        self.job_types = JobTypeCollection.from_config(self.config, env=self.env)
        self.processes = {}
        self.rng = np.random.default_rng(self.config['seed'])
        self.rng_list = self.rng.spawn(len(self.notation.graph.nodes) + 2)
        self.rand_containers = [RandomContainer(rng,
                                                mean=self.config['processes']['mean'],
                                                std=self.config['processes']['std'],
                                                beta=self.config['arrivalProcess']['beta'])
                                for rng in self.rng.spawn(len(self.notation.graph.nodes) + 2)]

        self.logger = Logger(self.config['loggingRate'], self)

    def __repr__(self):
        cls = self.__class__.__name__
        return f'{cls}(config={self.config!r}, notation={self.notation!r}, env={self.env!r})'

    def build_processes(self):
        if not self.notation.graph:
            raise Exception('Can not build system with no graph specified.')

        nodes = reversed(list(self.notation.graph.nodes(data=True)))

        for node, props in nodes:
            queue = Queue(props['data'], env=self.env)
            process = Process(node, queue=queue, rnd=self.rand_containers[node], env=self.env)
            self.processes[node] = process

            for _, out, props in self.notation.graph.edges(node, data=True):
                process.update_next({datum: self.processes[out] for datum in props['data']})

    def create_arrival_process(self):
        arrival_process = ArrivalProcess(-1, self.job_types, env=self.env, rnd=self.rand_containers[-2])
        self.processes[-1] = arrival_process

    def link_arrival_process(self):
        self.processes[-1].next = defaultdict(lambda: self.processes[0])

    def create_exit_process(self):
        last_process_id, _ = sorted(self.processes.items(), reverse=True)[0]

        queue = Queue(self.data, env=self.env)
        exit_process = ExitProcess(last_process_id + 1, queue=queue, rnd=self.rand_containers[-1], env=self.env)
        self.processes[last_process_id + 1] = exit_process

    def link_exit_process(self):
        last_process_id, last_process = sorted(self.processes.items(), reverse=True)[0]
        self.processes[last_process_id - 1].next = defaultdict(lambda: last_process)

    def build(self):
        if not self.notation:
            raise Exception('Can not build system with no notation specified.')

        self.build_processes()

        self.create_arrival_process()
        self.create_exit_process()
        self.link_arrival_process()
        self.link_exit_process()

        self.processes = dict(sorted(self.processes.items()))

    def run(self):
        self.env.process(self.logger.run())

        for _, process in self.processes.items():
            self.env.process(process.run())

        self.env.run(until=self.config.data['until'])


def main():
    pass


if __name__ == '__main__':
    main()
