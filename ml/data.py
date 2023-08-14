import pickle

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset


class ProcessDataset(Dataset):
    def __init__(self, da: xr.DataArray, scaling_factor: int = 1, reduction_factor: float = 0.0, offset: int = 1,
                 only_process: bool = False):
        self.da = da
        self.scaling_factor = scaling_factor
        self.reduction_factor = reduction_factor
        self.offset = offset
        self.only_process = only_process

        self.scale(self.scaling_factor)
        self.reduce(self.reduction_factor)

        if self.only_process:
            self.to_processes()

    @classmethod
    def from_path(cls, path: str, scaling_factor: int = 1, reduction_factor: float = 0.0, offset: int = 1,
                  only_process: bool = False):
        with open(path, 'rb') as f:
            da = pickle.load(f)
        return cls(da, scaling_factor, reduction_factor, offset, only_process)

    def scale(self, scaling_factor: int):
        assert 1 <= scaling_factor

        self.scaling_factor = scaling_factor
        self.da = self.da[::self.scaling_factor]

    def reduce(self, reduction_factor: float):
        assert 0.0 <= reduction_factor <= 1.0

        self.reduction_factor = reduction_factor
        limit = int((1 - self.reduction_factor) * len(self.da))
        self.da = self.da[:limit]

    def to_processes(self):
        self.da = xr.concat(self.da[::, 1::-2], dim='process')

    def get_sample_shape(self):
        return self[0][0].shape

    def __len__(self):
        return len(self.da) - self.offset

    def __getitem__(self, item):
        jobs = sorted(self.da['job'].data)

        dist_source = self.da.sel(job=jobs)[item].to_numpy()
        dist_target = self.da.sel(job=jobs)[item + self.offset].to_numpy()

        if self.only_process:
            return (torch.tensor(np.expand_dims(dist_source, axis=0), dtype=torch.float),
                    torch.tensor(np.expand_dims(dist_target, axis=0), dtype=torch.float))

        diff = dist_target - dist_source

        dist_source[0] = diff[0]
        dist_source[-1].fill(0)

        dist_target[0].fill(0)
        dist_target[-1] = diff[-1]

        return torch.tensor(dist_source, dtype=torch.float), torch.tensor(dist_target, dtype=torch.float)
