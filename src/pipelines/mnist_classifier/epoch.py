from kedro.io import DataCatalog, MemoryDataSet
from kedro.pipeline import pipeline
from kedro.pipeline.node import node

from src.common.const import SaverLoaderConst as slc
from src.nodes.common import output_concat, log_dict
from src.nodes.metrics import mse_metric
from src.nodes.save_load import save_state_dict, mlflow_registry
from src.nodes.test_train.mnist import test, one_epoch_mnist_classifier_train

epoch_data = DataCatalog(
    {
        "data_preprocess": MemoryDataSet(lambda x: x),
        "experiment_name": MemoryDataSet("test_2"),
        "checkpoint": MemoryDataSet(slc.LAST),
        "new_experiment": MemoryDataSet(True),
    }
)

epoch_pipeline = pipeline(
    [
        node(
            func=one_epoch_mnist_classifier_train,
            inputs=[
                "initialized_model",
                "initialized_optimizer",
                "loss",
                "train_data_loader",
                "device",
                "data_preprocess",
            ],
            outputs=[
                "train_time",
                "train_av_loss_metrics",
                "updated_optimizer",
                "updated_model",
            ],
        ),
        node(
            func=test,
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
            func=save_state_dict,
            inputs=["experiment_name", "metrics", "updated_model", "updated_optimizer", "epoch"],
            outputs="none_1",
        ),
        node(
            func=mlflow_registry,
            inputs={
                "experiment_name": "experiment_name",
                "metrics": "metrics",
                "model": "initialized_model",
                "optimizer": "initialized_optimizer",
                "device": "device",
                "loss": "loss",
                "epoch": "epoch",
                "lr": "lr"
            },
            outputs="none_2",
        ),
        node(
            func=log_dict,
            inputs={
                "metrics": "metrics",
                "epoch": "epoch",
            },
            outputs=["common_report", "metrics_report"]
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
