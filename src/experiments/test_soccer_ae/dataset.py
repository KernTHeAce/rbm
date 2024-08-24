import pandas as pd
import numpy as np

from torch.utils.data import Dataset


class SoccerCSVDataSet(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        self.data = []
        for index, row in df.iterrows():
            self.data.append(np.array(row, dtype=np.float64))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index_):
        return self.data[index_]
