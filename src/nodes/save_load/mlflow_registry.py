import mlflow

from src.common.const import MetricConst as mc
from src.common.const import CommonConst as cc
from src import MLRUNS_DIR


def mlflow_registry(experiment_name, epoch: int, metrics=None, new_experiment=True):
    mlflow.set_tracking_uri(MLRUNS_DIR)
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id
    except AttributeError:
        experiment_id = mlflow.create_experiment(experiment_name)

    if metrics:
        metrics_data = "   ".join([f"{key}: {round(item[mc.VALUE], 3)}" for key, item in metrics.items()])
    else:
        metrics_data = ""
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        mlflow.set_tag("mlflow.runName", f"Epoch: {epoch} {metrics_data}")
        for key, item in metrics.items():
            mlflow.log_metric(key, value=item[mc.VALUE])
    return cc.NONE


