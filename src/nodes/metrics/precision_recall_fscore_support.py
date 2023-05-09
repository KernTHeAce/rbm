from sklearn.metrics import precision_recall_fscore_support

from src.common.const import MetricConst as mc
from src.common.const import MetricsOutputValues as mov

from .tools import update_metrics


def per_label(y_true, y_pred, labels, metrics):
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true=y_true, y_pred=y_pred, labels=labels
    )
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
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true=y_true, y_pred=y_pred, average=param
    )
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
