import torch
import torchmetrics

from src import DEVICE


class base:
    def __call__(self, targets, outputs):
        return self.metric(torch.cat(targets).reshape(-1), torch.cat(outputs).reshape(-1)).item()


class f1(base):
    def __init__(self, num_classes):
        self.metric = torchmetrics.F1Score(task="multiclass", num_classes=num_classes).to(DEVICE)

    @property
    def __name__(self):
        return "f1"


class accuracy(base):
    def __init__(self, num_classes):
        self.metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(DEVICE)

    @property
    def __name__(self):
        return "accuracy"


class recall(base):
    def __init__(self, num_classes):
        self.metric = torchmetrics.Recall(task="multiclass", num_classes=num_classes).to(DEVICE)

    @property
    def __name__(self):
        return "recall"
