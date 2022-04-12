from torch.utils.data import Dataset, DataLoader

import numpy as np
import os
import torch

class PDEDataset(Dataset):
    def __init__(self, dataset_path):
        with open(dataset_path, 'rb') as f:
            self.data = torch.from_numpy(np.load(f))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def load_data(dataset_path, num_workers=0, batch_size=128):
    dataset = PDEDataset(dataset_path)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)

def error(pred, data):
    return np.linalg.norm(pred - data)