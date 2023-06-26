from __future__ import annotations

import math
from typing import TYPE_CHECKING

from matplotlib import pyplot as plt

from system.job import Job
if TYPE_CHECKING:
    from system.system import System


class Logger:
    def __init__(self, rate: float, system: System):
        self.rate = rate
        self.system = system
        self.log = {}

        self.default_dist = {job_type.name: 0 for job_type in self.system.job_types.types}

    def get_job_dist(self, jobs: [Job]):
        dist = dict(self.default_dist)
        for job in jobs:
            dist[job.type.name] += 1
        return dist

    def log_processes(self):
        job_dists = {}
        for _, process in self.system.processes.items():
            if process.queue:
                job_dists[process.id] = self.get_job_dist(process.queue.items)
        self.log[round(self.system.env.now, 1)] = job_dists

    def process(self):
        self.log_processes()

        yield self.system.env.timeout(self.rate)

    def run(self):
        while True:
            yield from self.process()

    def get_log_process_data(self, process: int):
        process_data = {}
        for t in self.log.keys():
            process_data[t] = sum(self.log[t][process].values())
        return process_data

    def get_log_processes_data(self):
        data = {}
        if not len(self.log):
            return data

        processes = self.log[0].keys()

        for process in processes:
            data[process] = self.get_log_process_data(process)
        return data

    def plot(self):
        data = self.get_log_processes_data()

        ncols = 4
        nrows = math.ceil(len(data) / ncols)
        fig, axs = plt.subplots(figsize=(16, 9), ncols=ncols, nrows=nrows)
        for queue_data, ax in zip(data.values(), axs.ravel()):
            ax.plot(queue_data.keys(), queue_data.values())

        plt.show()
