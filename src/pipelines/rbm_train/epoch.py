from kedro.io import DataCatalog, MemoryDataSet
from kedro.pipeline import pipeline
from kedro.pipeline.node import node

from src import BASE_DIR
from src.common.const import SaverLoaderConst as slc
from src.nodes.common import output_concat
from src.nodes.metrics import mse_metric
from src.nodes.save_load import save_state_dict
from src.nodes.test_train import one_epoch_ae_train, test

epoch_data = DataCatalog(
    {
        "data_preprocess": MemoryDataSet(lambda x: x),
        "experiment_path": MemoryDataSet(f"{BASE_DIR}/experiments/test_1"),
        "checkpoint": MemoryDataSet(slc.LAST),
        "new_experiment": MemoryDataSet(True),
    }
)

epoch_pipeline = pipeline(
    [
        node(
            func=one_epoch_ae_train,
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
                "data_preprocess",
                "train_av_loss_metrics",
            ],
            outputs=["test_time", "test_train_av_loss_metrics", "y_true", "y_pred"],
        ),
        node(func=mse_metric, inputs=["y_true", "y_pred", "test_train_av_loss_metrics"], outputs="metrics"),
        node(
            func=save_state_dict,
            inputs=["experiment_path", "metrics", "updated_model", "updated_optimizer", "epoch"],
            outputs="metrics_report",
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
