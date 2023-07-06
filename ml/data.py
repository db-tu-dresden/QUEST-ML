import pandas as pd
import torch
from torch.utils.data import Dataset


class ProcessDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df

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
        return len(self.df) - 1     # -1 because the next step is the label; therefore the last step is omitted

    def __getitem__(self, item):
        _, *dist_source = self.df.iloc[[item]].to_numpy()[0]
        _, *dist_target = self.df.iloc[[item + 1]].to_numpy()[0]
        return torch.tensor(dist_source, dtype=torch.float), torch.tensor(dist_target, dtype=torch.float)
