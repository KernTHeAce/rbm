import mlflow

from src import MLRUNS_DIR
from src.common.const import CommonConst as cc
from src.common.const import MetricConst as mc


def mlflow_registry(**kwargs):
    mlflow.set_tracking_uri(MLRUNS_DIR)
    try:
        experiment = mlflow.get_experiment_by_name(kwargs["experiment_name"])
        experiment_id = experiment.experiment_id
    except AttributeError:
        experiment_id = mlflow.create_experiment(kwargs["experiment_name"])
    mlflow.set_experiment(kwargs["experiment_name"])
    with mlflow.start_run():
        mlflow.set_tag("mlflow.runName", f"Epoch: {kwargs['epoch']}")
        for key, item in kwargs.items():
            if key in ["experiment_name", "epoch"]:
                continue
            elif key == "metrics":
                for metric_key, metric_item in kwargs["metrics"].items():
                    mlflow.log_metric(metric_key, value=metric_item[mc.VALUE])
            else:
                if key == "optimizer":
                    item = str(item).split()[0]
                mlflow.log_param(key, value=str(item))

    return cc.NONE
