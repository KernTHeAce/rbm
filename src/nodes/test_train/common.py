import pandas as pd
import torch
from torch.utils.data import DataLoader

from .datasets import CSVSoccerDataset


def csv_to_data_loader(csv_file: pd.DataFrame, batch_size: int = 1, shuffle: bool = False):
    dataset = CSVSoccerDataset(csv_file)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def dataset_to_dataloader(dataset, batch_size: int = 1, shuffle: bool = False):
    return torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
