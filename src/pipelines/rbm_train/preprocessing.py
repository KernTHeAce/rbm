from kedro.extras.datasets.pandas import CSVDataSet
from kedro.io import DataCatalog, MemoryDataSet
from kedro.pipeline import pipeline
from kedro.pipeline.node import node

from src import DATA_DIR, EXPERIMENTS_DIR
from src.common.const import SaverLoaderConst as slc
from src.nodes import output_concat
from src.nodes.save_load import load_state_dict
from src.nodes.test_train import csv_to_data_loader, get_ae_model, get_device, get_loss, get_optimizer, rbm_init_ae

preprocessing_data = DataCatalog(
    {
        "soccer_train_dataset": CSVDataSet(filepath=f"{DATA_DIR}/01_raw/wiscout/train_x_sigm_1221.csv"),
        "soccer_test_dataset": CSVDataSet(filepath=f"{DATA_DIR}/01_raw/wiscout/test_x_sigm_136.csv"),
        "train_data_loader": MemoryDataSet(copy_mode="assign"),
        "batch_size": MemoryDataSet(16),
        "shuffle": MemoryDataSet(True),
        "features": MemoryDataSet([39, 34, 29, 24, 19]),
        "is_cuda": MemoryDataSet(True),
        "lr": MemoryDataSet(1e-3),
        "experiment_path": MemoryDataSet(f"{EXPERIMENTS_DIR}/test_1"),
        "checkpoint": MemoryDataSet(slc.LAST),
        "new_experiment": MemoryDataSet(True),
    }
)

preprocessing_pipeline = pipeline(
    [
        node(
            func=load_state_dict,
            inputs=["experiment_path", "checkpoint", "new_experiment"],
            outputs=["loaded_model", "loaded_optimizer", "is_model_initialized"],
        ),
        node(
            func=csv_to_data_loader,
            inputs=["soccer_train_dataset", "batch_size", "shuffle"],
            outputs="train_data_loader",
        ),
        node(
            func=csv_to_data_loader,
            inputs=["soccer_test_dataset", "batch_size", "shuffle"],
            outputs="test_data_loader",
        ),
        node(func=get_device, inputs="is_cuda", outputs="device"),
        node(func=get_ae_model, inputs=["features", "loaded_model"], outputs="model"),
        node(func=get_loss, inputs=None, outputs="loss"),
        node(func=get_optimizer, inputs=["model", "lr"], outputs="initialized_optimizer"),
        node(
            func=rbm_init_ae,
            inputs=["model", "train_data_loader", "device", "is_model_initialized"],
            outputs="initialized_model",
        ),
        node(
            func=output_concat,
            inputs={
                "initialized_model": "initialized_model",
                "initialized_optimizer": "initialized_optimizer",
                "device": "device",
                "train_data_loader": "train_data_loader",
                "test_data_loader": "test_data_loader",
                "loss": "loss",
            },
            outputs="results",
        ),
    ]
)
