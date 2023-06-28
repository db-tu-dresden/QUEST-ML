from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pandas as pd
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
        self._df = pd.DataFrame()

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

    @property
    def df(self):
        steps = list(self.log.keys())

        if not steps:
            return self._df

        data = {'step': steps}

        processes = self.log[0].keys()
        for process in processes:
            process_data = []
            for t in self.log.keys():
                process_data.append(self.log[t][process])
            data[process] = process_data
        self._df = pd.DataFrame(data)

        return self._df

    def plot(self):
        df = self.df
        x = df.loc[:, 'step']

        ncols = 4
        nrows = math.ceil((len(df.columns) - 1) / ncols)
        fig, axs = plt.subplots(figsize=(16, 9), ncols=ncols, nrows=nrows)

        for (name, values), ax in zip(df.loc[:, df.columns != 'step'].items(), axs.ravel()):
            ax.plot(x, [sum(v for v in entry.values()) for entry in values])

        plt.show()
