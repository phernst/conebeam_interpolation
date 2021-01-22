import numpy as np
import torch


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets
        self.cslen = np.concatenate([
            [0],
            np.cumsum([len(d) for d in datasets])])

    def __len__(self):
        return self.cslen[-1]

    def __getitem__(self, idx):
        ds_idx = np.searchsorted(self.cslen - 1, idx) - 1
        pos_idx = idx - self.cslen[ds_idx]
        return self.datasets[ds_idx][pos_idx]
