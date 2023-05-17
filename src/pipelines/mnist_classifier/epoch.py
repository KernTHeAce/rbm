from kedro.pipeline import pipeline
from kedro.pipeline.node import node

from nodes import metrics
from nodes import save_load as sl
from nodes.common import log_dict, output_concat
from nodes.test_train import mnist

epoch_pipeline = pipeline(
    [
        node(
            func=mnist.train_mnist_classifier,
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
                "metrics_1",
                "updated_optimizer",
                "updated_model",
            ],
        ),
        node(
            func=mnist.test_mnist_classifier,
            inputs=[
                "initialized_model",
                "loss",
                "test_data_loader",
                "device",
                "preprocessing",
                "metrics_1",
            ],
            outputs=["test_time", "metrics_2", "y_true", "y_pred"],
        ),
        # node(func=metrics.mse_metric, inputs=["y_true", "y_pred", "metrics_2"], outputs="metrics_3"),
        node(func=metrics.average, inputs=["y_true", "y_pred", "metrics_2"], outputs="metrics"),
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
