import torch
from torchmetrics import MeanSquaredError

from src import DEVICE


def mse(inputs, outputs):
    metric = MeanSquaredError().to(DEVICE)
    return metric(torch.cat(inputs).reshape(-1), torch.cat(outputs).reshape(-1)).item()
