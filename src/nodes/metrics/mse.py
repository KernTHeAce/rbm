from typing import Any, Dict

from src.common.const import MetricConst as mc
from src.common.const import MetricsOutputValues as mov

from .tools import data_preprocess, update_metrics

import torch
import torch.nn as nn


def mse_metric(y_true, y_pred, metrics: Dict[str, Any] = None) -> Dict[str, Dict[str, Any]]:
    mse = nn.MSELoss()
    output = mse(torch.cat(y_pred), torch.cat(y_true))
    data = {
        mov.MSE: {
            mc.VALUE: output.item(),
            mc.BEST: mc.DOWN,
        },
    }
    return update_metrics(metrics, data)
