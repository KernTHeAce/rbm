import mlflow

from src import MLRUNS_DIR
from src.common.const import CommonConst as cc
from src.common.const import MetricConst as mc

mlflow.set_tracking_uri(MLRUNS_DIR)

def get_id_by_name(df, name):
    res = df.index[df["tags.mlflow.runName"] == name]
    if not len(res):
        return None
    return df["run_id"][res.item()]


def mlflow_registry(**kwargs):
    name = kwargs["experiment_name"]
    epoch = kwargs['epoch']
    runs_names = mlflow.search_runs(experiment_names=["Default"])
    if not len(runs_names):
        run_id = None
    else:
        run_id = get_id_by_name(runs_names, name)
    with mlflow.start_run(
            run_name=name,
            run_id=run_id,
            experiment_id='0',
    ):
        for key, item in kwargs.items():
            if key in ["experiment_name", "epoch"]:
                continue
            elif key == "metrics":
                for metric_key, metric_item in kwargs["metrics"].items():
                    mlflow.log_metric(metric_key, value=metric_item[mc.VALUE], step=epoch)
    return cc.NONE


def mlflow_registry_deprecated(**kwargs):
    mlflow.set_tracking_uri(MLRUNS_DIR)
    try:
        experiment = mlflow.get_experiment_by_name(kwargs["experiment_name"])
        experiment_id = experiment.experiment_id
    except AttributeError:
        experiment_id = mlflow.create_experiment(kwargs["experiment_name"])
        mlflow.set_tag("mlflow.runName", kwargs["experiment_name"])
        # mlflow.set_experiment(kwargs["experiment_name"])
    with mlflow.start_run():
        # mlflow.set_tag("mlflow.runName", f"Epoch: {kwargs['epoch']}")
        epoch = kwargs['epoch']
        for key, item in kwargs.items():
            if key in ["experiment_name", "epoch"]:
                continue
            elif key == "metrics":
                for metric_key, metric_item in kwargs["metrics"].items():
                    mlflow.log_metric(metric_key, value=metric_item[mc.VALUE], step=epoch)
            # else:
            #     if key == "optimizer":
            #         item = str(item).split()[0]
            #     mlflow.log_param(key, value=str(item))
    return cc.NONE
