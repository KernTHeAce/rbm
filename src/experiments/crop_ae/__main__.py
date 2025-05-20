import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from torch.utils.data import random_split
from src import BATCH_SIZE, DATA_DIR, DEVICE
from custom.rbm import generate_combinations, run_experiment
from core.metrics import MetricCalculator, regression
from core.models import BaseModel
from src.experiments import INITIALIZERS


class CustomDataset(Dataset):
    def __init__(self, csv_file, label_col):
        self.data = pd.read_csv(csv_file)

        yes_no_cols = self.data.select_dtypes(include=['object']).columns
        for col in yes_no_cols:
            if set(self.data[col].unique()) <= {"Yes", "No"}:
                self.data[col] = self.data[col].map({"Yes": 1, "No": 0})
        features_data = self.data.drop(columns=[label_col])
        numeric_cols = features_data.select_dtypes(include=['float']).columns
        features_numeric = features_data[numeric_cols].values.astype('float32')
        bool_cols = features_data.select_dtypes(include=['int']).columns
        features_bool = features_data[bool_cols].values.astype('float32')
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.features = torch.tensor(
            np.concatenate([features_numeric, features_bool], axis=1), dtype=torch.float64
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.features[idx]


dataset = CustomDataset(
    csv_file=f"{DATA_DIR}/crop/Crop_recommendation.csv",
    label_col="label"
)


train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


lengths = {"s": [2], "m": [5, 3, 5], "l": [7, 5, 3, 5, 7]}
model_combinations = generate_combinations(
    {
        "l": ["s", "m", "l"],
        "w_k": [1, 7, 15],
    }
)
MODEL_INPUT_SIZE = 7
metrics_calculator = MetricCalculator([regression.mae()])

for model_params in model_combinations:
    model = BaseModel(
        [MODEL_INPUT_SIZE]
        + [item * model_params["w_k"] for item in lengths[model_params["l"]]]
        + [MODEL_INPUT_SIZE]
    ).to(DEVICE)

    for initializer_type, value in INITIALIZERS.items():
        for initializer_params in value:
            response = run_experiment(
                test_loader,
                train_loader,
                f"crop_ae_l={model_params['l']}_wk={model_params['w_k']}",
                model,
                torch.nn.MSELoss(),
                initializer_params,
                initializer_type,
                metrics_calculator,
            )
