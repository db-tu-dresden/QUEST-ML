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

        self.da = self.da[::scaling_factor]
        self.da = self.da[:int((1 - self.reduction_factor) * len(self.da))]

        if self.only_process:
            self.da = xr.concat(self.da[::, 1::-2], dim='process')

    @classmethod
    def from_path(cls, path: str, scaling_factor: int = 1, reduction_factor: float = 0.0, offset: int = 1,
                  only_process: bool = False):
        with open(path, 'rb') as f:
            da = pickle.load(f)
        return cls(da, scaling_factor, reduction_factor, offset, only_process)

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
