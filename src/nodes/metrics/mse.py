from typing import Any, Dict

from src.common.const import MetricConst as mc
from src.common.const import MetricsOutputValues as mov

from .tools import data_preprocess, update_metrics
from torchmetrics import MeanSquaredError

import torch
import torch.nn as nn


def mse_metric(y_true, y_pred, metrics: Dict[str, Any] = None) -> Dict[str, Dict[str, Any]]:
    mse = MeanSquaredError()
    output = mse(torch.cat(y_pred).reshape(-1), torch.cat(y_true).reshape(-1))
    data = {
        mov.MSE: {
            mc.VALUE: output.item(),
            mc.BEST: mc.DOWN,
        },
    }
    return update_metrics(metrics, data)
