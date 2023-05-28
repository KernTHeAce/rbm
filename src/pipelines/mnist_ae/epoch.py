from kedro.pipeline import pipeline
from kedro.pipeline.node import node

from src.nodes import save_load as sl
from src.nodes.common import log_dict, output_concat
from src.nodes.metrics import mse_metric
from src.nodes.test_train import soccer as soc

epoch_pipeline = pipeline(
    [
        node(
            func=soc.train_ae,
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
            func=soc.test_ae,
            inputs=["updated_model", "loss", "test_data_loader", "device", "preprocessing", "train_av_loss_metrics"],
            outputs=["test_time", "test_train_av_loss_metrics", "y_true", "y_pred"],
        ),
        node(func=mse_metric, inputs=["y_true", "y_pred", "test_train_av_loss_metrics"], outputs="metrics"),
        node(
            func=sl.save_state_dict,
            inputs=["experiment_name", "metrics", "updated_model", "updated_optimizer", "epoch"],
            outputs=None,
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
            outputs=None,
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
