from __future__ import annotations

from system.environment import Environment
from system.job import Job, JobTypeCollection
from system.queue import Queue


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
