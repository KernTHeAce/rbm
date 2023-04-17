from typing import Dict, Any

from src.common.const import MetricsOutputValues as mov
from src.common.const import MetricConst as mc
import torch


def data_preprocess(data: list):
    result = None
    for item in data:
        if len(item.shape) == 2:
            for i in item:
                if result is None:
                    result = torch.clone(i)
                else:
                    torch.cat((result, i), 0)

    return result


def mse_metric(y_true, y_pred, metrics: Dict[str, Any]=None) -> Dict[str, Dict[str, Any]]:
    y_pred = data_preprocess(y_pred)
    y_true = data_preprocess(y_true)
    assert len(y_true) == len(y_pred)
    mse = sum([(y_1 - y_2) ** 2 for y_1, y_2 in zip(y_true, y_pred)]) / len(y_true)

    data = {
        mov.MSE: {
            mc.VALUE: mse.item(),
            mc.BEST: mc.DOWN,
        },
    }
    return data if metrics is None else metrics.update(data)
