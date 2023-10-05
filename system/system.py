from __future__ import annotations

import timeit
import typing
from collections import defaultdict

import numpy as np
import yaml

from notation import Notation
from system.config import Config
from system.environment import Environment
from system.job import JobTypeCollection, Job, JobType
from system.logger import Logger
from system.process import Process, ArrivalProcess, ExitProcess
from system.queue import Queue
from system.random import RandomContainer


class System:
    def __init__(self, config: Config, notation: Notation, env: Environment):
        super().__init__()
        if config is None or not isinstance(config, Config):
            raise ValueError(f'Config must be specified and of type Config. Got {config!r}')
        if notation is None or not isinstance(notation, Notation):
            raise ValueError(f'Notation must be specified and of type Notation. Got {notation!r}')
        config_data = set(job['name'] for job in config['jobs'])
        if config_data != notation.data.value:
            raise ValueError(f'Config and notation do not have the same data elements. '
                             f'Got {config_data!r} from config and {notation.data.value!r} from notation.')
        if len(config['processes']) != len(notation.graph.nodes):
            raise ValueError(f'Config and notation specify a different number of processes. '
                             f'Got {len(config["processes"])} processes from config and '
                             f'{len(notation.graph.nodes)} processes from notation.')

        self.config = config
        self.notation = notation
        self.data = [job['name'] for job in self.config['jobs']]

        self.until = None
        self.break_condition = None

        self.env = env
        env.system = self

        self.job_types = JobTypeCollection.from_config(self.config, env=self.env)
        self.processes = {}
        self.rng = np.random.default_rng(self.config['randomSeed'])
        self.rand_containers = self.build_rnd_containers()

        self.job_arrivals = self.load_job_arrivals(config['jobArrivalPath']) if config['jobArrivalPath'] else None

        self.logger = Logger(self.config['loggingRate'], self)

        self.build()

    def __repr__(self):
        cls = self.__class__.__name__
        return f'{cls}(config={self.config!r}, notation={self.notation!r}, env={self.env!r})'

    def set_break_condition(self, condition: typing.Callable[[ExitProcess], bool]):
        self.until = self.env.event()
        self.break_condition = condition

        exit_process: ExitProcess
        _, exit_process = sorted(self.processes.items())[-1]
        exit_process.break_event = self.until
        exit_process.break_condition = self.break_condition

    def build_rnd_containers(self):
        containers = dict()
        rngs = self.rng.spawn(len(self.notation.graph.nodes) + 1)

        for i, rng in enumerate(rngs[:-1]):
            key = self.config['processes'][i]['name'] or i
            containers[key] = RandomContainer(rng,
                                              mean=self.config['processes'][i]['mean'],
                                              std=self.config['processes'][i]['std'],
                                              beta=self.config['arrivalProcess']['beta'])

        # Exit Process
        containers[-1] = RandomContainer(rngs[-1])

        return containers

    @staticmethod
    def load_job_arrivals(path: str):
        with open(path) as f:
            return yaml.full_load(f)

    def build_processes(self):
        if not self.notation.graph:
            raise Exception('Can not build system with no graph specified.')

        nodes = reversed(list(self.notation.graph.nodes(data=True)))
        n = len(self.notation.graph.nodes)

        if n == 0:
            raise Exception('Can not build system with no nodes specified.')

        for i, (node, props) in enumerate(nodes):
            queue = Queue(props['data'], env=self.env)
            try:
                rand_container = self.rand_containers[props.get('name', node)]
            except KeyError:
                raise Exception(f'Notation and config process name missmatch. '
                                f'Got {props.get("name", node)} from notation, not found in config.')

            if i == n - 1:
                process = ArrivalProcess(-1, self.job_types,
                                         rnd=rand_container,
                                         env=self.env,
                                         job_arrivals=self.job_arrivals,
                                         name=props.get('name'),
                                         continue_with_rnd_jobs=self.config['continueWithRndJobs'])
            else:
                process = Process(node,
                                  queue=queue,
                                  rnd=rand_container,
                                  env=self.env,
                                  name=props.get('name'))
            self.processes[node] = process

            for _, out, props in self.notation.graph.edges(node, data=True):
                process.update_next({datum: self.processes[out] for datum in props['data']})

    def create_exit_process(self):
        last_process_id, _ = sorted(self.processes.items(), reverse=True)[0]

        queue = Queue(self.data, env=self.env)
        exit_process = ExitProcess(last_process_id + 1,
                                   queue=queue,
                                   rnd=self.rand_containers[-1],
                                   env=self.env,
                                   name='Exit Process')
        self.processes[last_process_id + 1] = exit_process

    def link_exit_process(self):
        last_process_id, last_process = sorted(self.processes.items(), reverse=True)[0]
        self.processes[last_process_id - 1].next = defaultdict(lambda: last_process)

    def build(self):
        if not self.notation:
            raise Exception('Can not build system with no notation specified.')

        self.build_processes()

        self.create_exit_process()
        self.link_exit_process()

        self.processes = dict(sorted(self.processes.items()))

    def run(self, until=None):
        start = timeit.default_timer()

        self.env.process(self.logger.run())

        for _, process in self.processes.items():
            self.env.process(process.run())

        self.env.run(until=until or self.until or self.config['until'])

        stop = timeit.default_timer()

        self.logger.runtime = stop - start

    def set_state(self, state: np.array):
        assert state.shape == (len(self.processes), len(self.job_types.types))

        process: Process
        for i, process in self.processes.items():
            dist = state[i]
            job_type: JobType
            for j, job_type in enumerate(self.job_types.types):
                amount = dist[j]
                for k in range(amount):
                    job = Job(-1, job_type, env=self.env)
                    process.push(job)
