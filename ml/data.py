import pickle
import random

import numpy
import torch
import xarray as xr
from torch.utils.data import Dataset

from ml import Config


class ProcessDataset(Dataset):
    def __init__(self, da: xr.DataArray, scaling_factor: int = 1, reduction_factor: float = 0.0, offset: int = 1,
                 only_process: bool = False, enhances: int = 0, base_lambda: float = 1.0,
                 lambda_variability: float = 0.1, accumulation_window: int = 1):
        self.da = da
        self.scaling_factor = scaling_factor
        self.reduction_factor = reduction_factor
        self.offset = offset
        self.only_process = only_process
        self.accumulation_window = accumulation_window

        self.scale(self.scaling_factor)
        self.reduce(self.reduction_factor)

        source_da, target_da = self.get_offset_das()

        if self.only_process:
            source_da = self.to_processes(source_da)
            target_da = self.to_processes(target_da)
        else:
            source_da, target_da = self.set_accumulation_window(source_da, target_da, self.accumulation_window)

        self.das = [{'source': source_da, 'target': target_da}]
        self.enhance(enhances, base_lambda, lambda_variability)

    def get_offset_das(self):
        jobs = sorted(self.da['job'].data)

        source_da = self.da.sel(job=jobs).copy()
        target_da = self.da.sel(job=jobs).copy()

        if self.offset == 0:
            return source_da, target_da

        source_da = source_da[:-abs(self.offset)]
        target_da = target_da[abs(self.offset):]

        if self.offset < 0:
            return target_da, source_da
        return source_da, target_da

    def set_accumulation_window(self, source_da, target_da, window_size):
        assert window_size >= 0

        if window_size == 0:
            return source_da, target_da

        if self.offset < 0:
            source_da, target_da = target_da, source_da

        reducer = None
        curr_elem = None
        prev_elem = None

        for i, (source_elem, target_elem) in enumerate(zip(source_da, target_da)):
            prev_elem = curr_elem
            curr_elem = source_elem.copy()

            if i % window_size == 0:
                reducer = source_elem.copy()

            if prev_elem is not None:
                source_da[i - 1][0] = source_elem[0] - prev_elem[0]
            source_elem[-1] -= reducer[-1]

            if prev_elem is not None:
                target_da[i - 1][0] = target_elem[0] - prev_elem[0]
            target_elem[-1] -= reducer[-1]

        source_da[-1][0] = numpy.zeros(source_da[-1][0].shape)
        target_da[-1][0] = numpy.zeros(target_da[-1][0].shape)

        if self.offset < 0:
            return target_da, source_da
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
    def from_config(cls, path: str, config: Config):
        if config['verbose']:
            print(f'Load data array from path: {path}')

        with open(path, 'rb') as f:
            da = pickle.load(f)

        if config['verbose']:
            print(f'Process data array...')

        return cls(da,
                   config['scaling_factor'],
                   config['reduction_factor'],
                   config['offset'],
                   config['only_process'],
                   config['enhances'],
                   config['enhance_base_lambda'],
                   config['enhance_lambda_variability'],
                   config['accumulation_window'])

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
        return xr.concat(da[::, 1::-2], dim='process').expand_dims('place_holder', axis=1)

    def get_sample_shape(self):
        return self[0][0].shape

    def __len__(self):
        return len(self.das) * len(self.das[0]['source'])

    def __getitem__(self, item):
        i = item // len(self.das[0]['source'])
        item = item % len(self.das[0]['source'])

        return (torch.tensor(self.das[i]['source'][item].to_numpy(), dtype=torch.float),
                torch.tensor(self.das[i]['target'][item].to_numpy(), dtype=torch.float))
