from torch.utils.data import Dataset, DataLoader

import numpy as np
import os
import torch

class PDEDataset(Dataset):
    def __init__(self, dataset_path, transforms = None, store_norms = False):
        self.store_norms = False
        with open(dataset_path, 'rb') as f:
            self.data = torch.from_numpy(np.load(f)).float()
            if store_norms:
                self.store_norms = store_norms
                self.norms = self.data.norm(dim = 1)
            if transforms:
                self.data = transforms(self.data)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.store_norms:
            return (self.data[idx], self.norms[idx])
        else:
            return self.data[idx]


def load_data(dataset_path, num_workers=0, batch_size=128, transforms = None, store_norms = False):
    dataset = PDEDataset(dataset_path, transforms = transforms, store_norms = store_norms)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)

def error(pred, data):
    return torch.norm(pred - data)