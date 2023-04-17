from dataclasses import dataclass


@dataclass(frozen=True)
class MetricAverageValues:
    BINARY: str = "binary"
    MICRO: str = "micro"
    MACRO: str = "macro"
    SAMPLES: str = "samples"
    WEIGHTED: str = "weighted"


@dataclass(frozen=True)
class MetricConst:
    AVERAGE: str = "average"
    PER_LABEL: str = "per_label"
    MSE: str = "mse"

    VALUE: str = "value"
    BEST: str = "best"
    UP: str = "up",
    DOWN: str = "down"


@dataclass(frozen=True)
class MetricsOutputValues:
    SUPPORT: str = "support"
    PRECISION: str = "precision"
    RECALL: str = "recall"
    F1: str = "f1"
    MSE: str = "mse"
