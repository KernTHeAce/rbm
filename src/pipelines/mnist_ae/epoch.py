from kedro.io import DataCatalog, MemoryDataSet
from kedro.pipeline import pipeline
from kedro.pipeline.node import node

from common.const import SaverLoaderConst as slc
from nodes import save_load as sl
from nodes.common import log_dict, output_concat
from nodes.metrics import mse_metric

from nodes.test_train import soccer as soc

epoch_data = DataCatalog(
    {
        "checkpoint": MemoryDataSet(slc.LAST),
        "new_experiment": MemoryDataSet(True),
    }
)


epoch_pipeline = pipeline(
    [
        node(
            func=soc.train_soccer_ae,
            inputs=[
                "initialized_model",
                "initialized_optimizer",
                "loss",
                "train_data_loader",
                "device",
                "preprocessing",
            ],
            outputs=[
                "train_time",
                "train_av_loss_metrics",
                "updated_optimizer",
                "updated_model",
            ],
        ),
        node(
            func=soc.test_soccer_ae,
            inputs=[
                "initialized_model",
                "loss",
                "test_data_loader",
                "device",
                "preprocessing",
                "train_av_loss_metrics",
            ],
            outputs=["test_time", "test_train_av_loss_metrics", "y_true", "y_pred"],
        ),
        node(func=mse_metric, inputs=["y_true", "y_pred", "test_train_av_loss_metrics"], outputs="metrics"),
        node(
            func=sl.save_state_dict,
            inputs=["experiment_name", "metrics", "updated_model", "updated_optimizer", "epoch"],
            outputs="none_1",
        ),
        node(
            func=sl.mlflow_registry,
            inputs={
                "experiment_name": "experiment_name",
                "metrics": "metrics",
                "model": "initialized_model",
                "optimizer": "initialized_optimizer",
                "device": "device",
                "loss": "loss",
                "epoch": "epoch",
                "lr": "lr",
            },
            outputs="none_2",
        ),
        node(
            func=log_dict,
            inputs={
                "metrics": "metrics",
                "epoch": "epoch",
            },
            outputs=["common_report", "metrics_report"],
        ),
        node(
            func=output_concat,
            inputs={
                "initialized_model": "updated_model",
                "initialized_optimizer": "updated_optimizer",
            },
            outputs="results",
        ),
    ]
)
