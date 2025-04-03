import torchmetrics

from src import DEVICE
from .classification import base


class mae(base):
    def __init__(self):
        self.metric = torchmetrics.MeanAbsoluteError().to(DEVICE)

    @property
    def __name__(self):
        return "mae"
