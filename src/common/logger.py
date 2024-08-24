import mlflow
from src import MLRUNS_DIR

mlflow.set_tracking_uri(MLRUNS_DIR)


class MlFlowLogger:
    def __init__(self, experiment_name: str, run_name):
        self.experiment = mlflow.set_experiment(experiment_name)
        self.run_name = run_name

    def get_run_id(self, name):
        runs_names = mlflow.search_runs(experiment_names=[self.experiment.name])
        if not len(runs_names):
            return None
        res = runs_names.index[runs_names["tags.mlflow.runName"] == name]
        if not len(res):
            return None
        return runs_names["run_id"][res.item()]

    def log_metrics(self, metrics, epoch):
        with mlflow.start_run(
            run_name=self.run_name,
            experiment_id=self.experiment.experiment_id,
            run_id=self.get_run_id(self.run_name)
        ):
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, value=metric_value, step=epoch)
