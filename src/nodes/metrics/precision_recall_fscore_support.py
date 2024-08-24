# from sklearn.metrics import precision_recall_fscore_support

from src.common.const import MetricConst as mc
from src.common.const import MetricsOutputValues as mov

from .tools import data_preprocess, update_metrics


def per_label(y_true, y_pred, labels, metrics):
    precision, recall, f1, support = precision_recall_fscore_support(y_true=y_true, y_pred=y_pred, labels=labels)
    data = {
        mov.PRECISION: {
            mc.VALUE: precision,
            mc.BEST: mc.UP,
        },
        mov.RECALL: {
            mc.VALUE: recall,
            mc.BEST: mc.UP,
        },
        mov.F1: {
            mc.VALUE: f1,
            mc.BEST: mc.UP,
        },
        mov.SUPPORT: {
            mc.VALUE: support,
            mc.BEST: mc.UP,
        },
    }
    return update_metrics(metrics, data)


def average(y_true, y_pred, metrics, param: str = "weighted"):
    y_true = data_preprocess(y_true)
    y_pred = data_preprocess(y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true=y_true, y_pred=y_pred, average=param)
    data = {
        mov.PRECISION: {
            mc.VALUE: precision.item(),
            mc.BEST: mc.UP,
        },
        mov.RECALL: {
            mc.VALUE: recall.item(),
            mc.BEST: mc.UP,
        },
        mov.F1: {
            mc.VALUE: f1.item(),
            mc.BEST: mc.UP,
        },
    }
    return update_metrics(metrics, data)
