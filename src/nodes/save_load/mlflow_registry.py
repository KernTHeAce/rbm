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


params = ["rbm_epoch", "rbm_init_type", "rbm_type", "lr"]


def mlflow_registry(**kwargs):
    name = kwargs["experiment_name"]
    epoch = kwargs["epoch"]
    runs_names = mlflow.search_runs(experiment_names=["Default"])
    if not len(runs_names):
        run_id = None
    else:
        run_id = get_id_by_name(runs_names, name)
    with mlflow.start_run(
        run_name=name,
        run_id=run_id,
        experiment_id="0",
    ):
        if run_id is None:
            for param in params:
                mlflow.log_param(param, kwargs[param])
        for key, item in kwargs.items():
            if key in ["experiment_name", "epoch"]:
                continue
            elif key == "metrics":
                for metric_key, metric_item in kwargs["metrics"].items():
                    mlflow.log_metric(metric_key, value=metric_item[mc.VALUE], step=epoch)
    return cc.NONE
