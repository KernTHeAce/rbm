from kedro.pipeline import pipeline
from kedro.pipeline.node import node

from src.nodes import metrics
from src.nodes import save_load as sl
from src.nodes.common import log_dict, output_concat
from src.nodes.test_train import mnist

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
                "epoch": "epoch",
                "lr": "lr",
                "rbm_epoch": "rbm_epoch",
                "rbm_init_type": "rbm_init_type",
                "rbm_type": "rbm_type",
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
