import pickle

import torch
from torch.utils.data import Dataset
import xarray as xr


class ProcessDataset(Dataset):
    def __init__(self, da: xr.DataArray, scaling_factor: int = 1, prediction_gap: int = 1):
        self.da = da
        self.scaling_factor = scaling_factor
        self.offset = prediction_gap

        self.orig_da = self.da
        self.da = self.da[::scaling_factor]

    @classmethod
    def from_path(cls, path: str, scaling_factor: int = 1):
        with open(path, 'rb') as f:
            da = pickle.load(f)
        return cls(da, scaling_factor)

    def get_sample_shape(self):
        return self[0][0].shape

    def __len__(self):
        return len(self.da) - self.offset

    def __getitem__(self, item):
        jobs = sorted(self.da['job'].data)

        diff = self.da.sel(job=jobs)[item + self.offset].to_numpy() - self.da.sel(job=jobs)[item].to_numpy()

        dist_source = self.da.sel(job=jobs)[item].to_numpy()
        dist_source[0] = diff[0]
        dist_source[-1].fill(0)

        dist_target = self.da.sel(job=jobs)[item + self.offset].to_numpy()
        dist_target[0].fill(0)
        dist_target[-1] = diff[-1]

        return torch.tensor(dist_source, dtype=torch.float), torch.tensor(dist_target, dtype=torch.float)
