import pandas as pd
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

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        return self.df.iloc[[item]]


def main():
    ProcessDataset.from_path('../save/df.pkl')


if __name__ == '__main__':
    main()
