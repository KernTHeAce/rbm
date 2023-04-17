import numpy as np
from torch.utils.data import Dataset


class CSVSoccerDataset(Dataset):
    def __init__(self, price_df):
        self.x_train = []
        for index, row in price_df.iterrows():
            self.x_train.append(np.array(row, dtype=np.float64))

    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, idx):
        return self.x_train[idx]
