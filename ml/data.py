import pandas as pd
import torch
from torch.utils.data import Dataset


class ProcessDataset(Dataset):
    def __init__(self, df: pd.DataFrame, scaling_factor=10):
        self.df = df
        self.scaling_factor = scaling_factor

        for name, data in self.df.items():
            if name == 'step':
                continue
            self.df[name] = data.map(lambda x: list(x.values()))

    @classmethod
    def from_path(cls, path: str):
        return cls(pd.read_pickle(path))

    def get_sample_shape(self):
        return self[0][0].shape

    def __len__(self):
        return len(self.df) // self.scaling_factor - 1     # -1 because the next step is the label; therefore the last step is omitted

    def __getitem__(self, item):
        _, *dist_source = self.df.iloc[[item * self.scaling_factor]].to_numpy()[0]
        _, *dist_target = self.df.iloc[[(item + 1) * self.scaling_factor]].to_numpy()[0]
        return torch.tensor(dist_source, dtype=torch.float), torch.tensor(dist_target, dtype=torch.float)
