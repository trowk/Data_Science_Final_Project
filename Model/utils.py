from torch.utils.data import Dataset, DataLoader

import numpy as np
import os
import torch

class PDEDataset(Dataset):
    def __init__(self, dataset_path, transforms = None, store_norms = False, noise=None):
        self.store_norms = False
        self.noise = noise
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
        n = torch.zeros(20)
        if self.noise and self.noise != 0:
            n = torch.normal(torch.zeros(20), torch.ones(20))
            n /= (n.norm() * self.noise)
        if self.store_norms:
            return (self.data[idx] + n, self.norms[idx])
        else:
            return self.data[idx] + n

class SyntheticData(Dataset):
    def __init__(self, mean=0, std=1, normalize=True, data_len = 10000, store_data = False):
        self.mean = float(mean)
        self.std = float(std)
        self.normalize = normalize
        self.store_data = store_data
        self.data_len = data_len
        self.data = None
        if store_data:
            self.data = torch.normal(torch.full((data_len, 20), self.mean), torch.full((data_len, 20), self.std))
            if self.normalize:
                self.data = torch.nn.functional.normalize(self.data)


    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        if self.store_data:
            return self.data[idx]
        else:
            d = torch.normal(torch.full((20,), self.mean), torch.full((20,), self.std))
            if self.normalize:
                d /= d.norm()
            return d

def load_data(dataset_path, num_workers=0, batch_size=128, transforms = None, store_norms = False, noise=None):
    dataset = PDEDataset(dataset_path, transforms = transforms, store_norms = store_norms, noise=noise)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)

def load_synthetic(num_workers=0, batch_size=128, mean=0, std=1, normalize=True, data_len = 10000, store_data = False):
    dataset = SyntheticData(mean=mean, std=std, normalize=normalize, data_len = data_len, store_data = store_data)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)

def error(pred, data):
    return torch.norm(pred - data)