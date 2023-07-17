import pandas as pd
import torch
from torch.utils.data import Dataset


class ProcessDataset(Dataset):
    def __init__(self, df: pd.DataFrame, scaling_factor: int = 10):
        self.df = df
        self.scaling_factor = scaling_factor

        for name, data in self.df.items():
            if name == 'step':
                continue
            self.df[name] = data.map(lambda x: list(x.values()))

        self.orig_df = self.df
        self.df = pd.DataFrame(self.df.iloc[::self.scaling_factor, :]).reset_index(drop=True)

    @classmethod
    def from_path(cls, path: str):
        return cls(pd.read_pickle(path))

    def get_sample_shape(self):
        return self[0][0].shape

    def __len__(self):
        return len(self.df) - 1     # -1 because the next step is the label; therefore the last step is omitted

    def __getitem__(self, item):
        _, *dist_source = self.df.iloc[[item]].to_numpy()[0]
        _, *dist_target = self.df.iloc[[item]].to_numpy()[0]
        return torch.tensor(dist_source, dtype=torch.float), torch.tensor(dist_target, dtype=torch.float)
