import pickle
import random

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset


class ProcessDataset(Dataset):
    def __init__(self, da: xr.DataArray, scaling_factor: int = 1, reduction_factor: float = 0.0, offset: int = 1,
                 only_process: bool = False, enhances: int = 0, base_lambda: float = 1.0,
                 lambda_variability: float = 0.1):
        self.da = da
        self.scaling_factor = scaling_factor
        self.reduction_factor = reduction_factor
        self.offset = offset
        self.only_process = only_process

        self.scale(self.scaling_factor)
        self.reduce(self.reduction_factor)

        source_da, target_da = self.get_offset_das()

        if self.only_process:
            source_da = self.to_processes(source_da)
            target_da = self.to_processes(target_da)
        else:
            source_da, target_da = self.set_diff(source_da, target_da)

        self.das = [{'source': source_da, 'target': target_da}]
        self.enhance(enhances, base_lambda, lambda_variability)

    def get_offset_das(self):
        jobs = sorted(self.da['job'].data)

        source_da = self.da.sel(job=jobs).copy()
        target_da = self.da.sel(job=jobs).copy()

        if self.offset == 0:
            return source_da, target_da
        return source_da[:-self.offset], target_da[self.offset:]

    def set_diff(self, source_da, target_da):
        for i in range(len(source_da)):
            diff = target_da[i] - source_da[i]

            source_da[i][0] = diff[0]
            source_da[i][-1] = np.zeros(source_da[i][-1].shape)

            target_da[i][0] = np.zeros(target_da[i][0].shape)
            target_da[i][-1] = diff[-1]

        return source_da, target_da

    def enhance(self, n: int, base_lambda: float, lambda_variability: float):
        source_da = self.das[0]['source']
        target_da = self.das[0]['target']

        for _ in range(1, n + 1):
            _s = source_da.copy()
            _t = target_da.copy()

            _lambda = base_lambda + random.uniform(-lambda_variability, lambda_variability)
            job_count = len(_s[0])

            for i in range(len(_s)):
                in_rand_dist = [int(random.expovariate(_lambda)) for _ in range(job_count)]

                if self.only_process:
                    _s[i] += in_rand_dist
                    _t[i] += in_rand_dist
                    continue

                out_rand_dist = [int(random.expovariate(_lambda)) for _ in range(job_count)]

                _s[i][0] += in_rand_dist
                _s[i][-1] += out_rand_dist

                _t[i][0] += in_rand_dist
                _t[i][-1] += out_rand_dist

            self.das.append({'source': _s, 'target': _t})

    @classmethod
    def from_path(cls, path: str, scaling_factor: int = 1, reduction_factor: float = 0.0, offset: int = 1,
                  only_process: bool = False, enhances: int = 0, base_lambda: float = 1.0,
                  lambda_variability: float = 0.1):
        with open(path, 'rb') as f:
            da = pickle.load(f)
        return cls(da, scaling_factor, reduction_factor, offset, only_process, enhances,
                   base_lambda, lambda_variability)

    def scale(self, scaling_factor: int):
        assert 1 <= scaling_factor

        self.scaling_factor = scaling_factor
        self.da = self.da[::self.scaling_factor]

    def reduce(self, reduction_factor: float):
        assert 0.0 <= reduction_factor <= 1.0

        self.reduction_factor = reduction_factor
        limit = int((1 - self.reduction_factor) * len(self.da))
        self.da = self.da[:limit]

    @staticmethod
    def to_processes(da):
        return xr.concat(da[::, 1::-2], dim='process')

    def get_sample_shape(self):
        return self[0][0].shape

    def __len__(self):
        return len(self.das) * len(self.das[0]['source'])

    def __getitem__(self, item):
        i = item // len(self.das[0]['source'])
        item = item % len(self.das[0]['source'])

        return (torch.tensor(self.das[i]['source'][item].to_numpy(), dtype=torch.float),
                torch.tensor(self.das[i]['target'][item].to_numpy(), dtype=torch.float))
