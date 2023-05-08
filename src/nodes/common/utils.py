from src import EXPERIMENTS_DIR
from src.common.const import MetricConst as mc


def output_concat(**kwargs):
    return kwargs


def log_dict(**kwargs):
    metrics_report = ""
    if mc.METRICS in kwargs:
        for metric_name, value in kwargs[mc.METRICS].items():
            metrics_report += f"{metric_name}: {value[mc.VALUE]}    "
    common_report = ""
    for key, value in kwargs.items():
        if key == mc.METRICS:
            continue
        common_report += f"{key}: {value}    "
    print(common_report + metrics_report)
    return common_report, metrics_report
