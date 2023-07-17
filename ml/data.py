import pickle

import torch
from torch.utils.data import Dataset
import xarray as xr


class ProcessDataset(Dataset):
    def __init__(self, da: xr.DataArray, scaling_factor: int = 10, window_size: int = 10):
        self.da = da
        self.scaling_factor = scaling_factor
        self.window_size = window_size

        self.orig_da = self.da
        self.da = self.da[::scaling_factor]

    @classmethod
    def from_path(cls, path: str):
        with open(path, 'rb') as f:
            da = pickle.load(f)
        return cls(da)

    def get_sample_shape(self):
        return self[0][0].shape

    def __len__(self):
        return len(self.da) - 1     # -1 because the next step is the label; therefore the last step is omitted

    def __getitem__(self, item):
        jobs = sorted(self.da['job'].data)

        dist_source = self.da.sel(job=jobs)[item].to_numpy()
        dist_target = self.da.sel(job=jobs)[item].to_numpy()

        return torch.tensor(dist_source, dtype=torch.float), torch.tensor(dist_target, dtype=torch.float)
