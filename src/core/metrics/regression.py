import torch
import torchmetrics

from src import DEVICE


def mse(targets, outputs):
    metric = torchmetrics.MeanSquaredError().to(DEVICE)
    return metric(torch.cat(targets).reshape(-1), torch.cat(outputs).reshape(-1)).item()
