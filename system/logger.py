from __future__ import annotations

import math
import pickle
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
from matplotlib import pyplot as plt

from system import ArrivalProcess

if TYPE_CHECKING:
    from system.system import System


class Logger:
    def __init__(self, rate: float, system: System):
        self.rate = rate
        self.system = system
        self.log = {}

        self.default_dist = {job_type.name: 0 for job_type in self.system.job_types.types}
        self._da = xr.DataArray()

    def __repr__(self):
        cls = self.__class__.__name__
        return f'{cls}(rate={self.rate!r}, system={self.system!r})'

    def log_processes(self):
        job_dists = {}
        for i, process in self.system.processes.items():
            job_dists[process.name] = dict(process.job_dist)
        self.log[round(self.system.env.now, 1)] = job_dists

    def process(self):
        self.log_processes()

        yield self.system.env.timeout(self.rate)

    def get_current_state(self):
        job_dists = {}
        for i, process in self.system.processes.items():
            job_dists[process.id] = dict(process.job_dist)
        return np.array([[job for job in process.values()] for process in job_dists.values()])

    def get_job_arrivals(self):
        arrival_process: ArrivalProcess
        _, arrival_process = sorted(self.system.processes)[0]
        return arrival_process.observed_job_arrivals

    def run(self):
        while True:
            yield from self.process()

    @property
    def da(self):
        steps = list(self.log.keys())
        processes = list(self.log[0].keys())
        jobs = list(self.log[0][processes[0]].keys())

        npa = np.array([[list(p.values()) for p in t.values()] for t in self.log.values()])
        self._da = xr.DataArray(data=npa, dims=['step', 'process', 'job'],
                                coords=dict(
                                    step=steps,
                                    process=processes,
                                    job=jobs,
                                ),
                                attrs=dict(
                                    granularity=self.rate),
                                )

        return self._da

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self.da, f)

    def plot(self, path: str = None, show: bool = True):
        ncols = 4
        nrows = math.ceil((len(self.da['process'])) / ncols)
        fig, axs = plt.subplots(figsize=(16, 9), ncols=ncols, nrows=nrows, layout='constrained')

        for process, ax in zip(self.da['process'].data, axs.ravel()):
            ax.plot(self.da['step'], self.da.sel(process=process).sum('job'))
            ax.set_title(f'Process {process}')

        if path:
            plt.savefig(path)
        if show:
            plt.show()
