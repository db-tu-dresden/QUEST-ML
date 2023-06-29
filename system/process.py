from __future__ import annotations

from system.environment import Environment
from system.job import Job, JobTypeCollection
from system.queue import Queue
from system.random import RandomContainer


class Process:
    def __init__(self, id: int, env: Environment, rnd: RandomContainer, queue: Queue = None):
        super().__init__()
        self.id = id
        self.queue = queue
        self.next = {}
        self.rng = rnd.rng
        self.mean = rnd.mean
        self.std = rnd.std
        self.beta = rnd.beta

        self.env = env

        self.job = None

    def update_next(self, update_dict: {str: Process}):
        self.next.update(update_dict)

    def push(self, job: Job):
        return self.queue.put(job)

    def process(self):
        job = yield self.queue.get()
        self.job = job

        yield job.service(self.rng, self.mean, self.std)

        self.job = None
        yield self.next[job.type.name].push(job)

    def run(self):
        while True:
            yield from self.process()

    def __repr__(self):
        cls = self.__class__.__name__
        return f'{cls}(id={self.id!r}, env={self.env!r}, rng={self.rng!r}, queue={self.queue!r})'


class ArrivalProcess(Process):
    def __init__(self, id: int, job_types: JobTypeCollection, env: Environment, rnd: RandomContainer):
        super().__init__(id, env, rnd)
        self.job_types = job_types
        self.last_job_id = -1

    def process(self):
        t = self.rng.exponential(scale=self.beta)
        yield self.env.timeout(t)

        self.last_job_id += 1
        job = self.job_types.get_rand_job(self.last_job_id, self.rng)

        yield self.next[job.type.name].push(job)

    def __repr__(self):
        cls = self.__class__.__name__
        return f'{cls}(id={self.id!r}, job_types={self.job_types!r}, env={self.env!r}, rng={self.rng!r})'


class ExitProcess(Process):
    def process(self):
        yield self.env.timeout(1)