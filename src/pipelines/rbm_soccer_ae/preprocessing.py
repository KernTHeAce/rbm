from kedro.extras.datasets.pandas import CSVDataSet
from kedro.io import DataCatalog, MemoryDataSet
from kedro.pipeline import pipeline
from kedro.pipeline.node import node

from nodes.test_train import common
from nodes.test_train import loss_optim_device as lod
from nodes.test_train import soccer as soc
from src import DATA_DIR, EXPERIMENTS_DIR
from src.common.const import SaverLoaderConst as slc
from src.nodes import output_concat
from src.nodes.save_load import load_state_dict

preprocessing_data = DataCatalog(
    {
        "soccer_train_dataset": CSVDataSet(filepath=f"{DATA_DIR}/soccer/01_raw/wiscout/train_x_sigm_1221.csv"),
        "soccer_test_dataset": CSVDataSet(filepath=f"{DATA_DIR}/soccer/01_raw/wiscout/test_x_sigm_136.csv"),
        "train_data_loader": MemoryDataSet(copy_mode="assign"),
        "batch_size": MemoryDataSet(16),
        "shuffle": MemoryDataSet(True),
        "features": MemoryDataSet([39, 34, 29, 24, 19]),
        "is_cuda": MemoryDataSet(False),
        "lr": MemoryDataSet(1e-3),
        "experiment_path": MemoryDataSet(f"{EXPERIMENTS_DIR}/test_1"),
        "checkpoint": MemoryDataSet(slc.LAST),
        "new_experiment": MemoryDataSet(True),
        "preprocessing": MemoryDataSet(lambda x: x),
    }
)

preprocessing_pipeline = pipeline(
    [
        node(
            func=load_state_dict,
            inputs=["experiment_path", "checkpoint", "new_experiment"],
            outputs=["loaded_model", "loaded_optimizer", "loaded_epoch", "is_model_initialized"],
        ),
        node(
            func=common.csv_to_data_loader,
            inputs=["soccer_train_dataset", "batch_size", "shuffle"],
            outputs="train_data_loader",
        ),
        node(
            func=common.csv_to_data_loader,
            inputs=["soccer_test_dataset", "batch_size", "shuffle"],
            outputs="test_data_loader",
        ),
        node(func=lod.get_device, inputs="is_cuda", outputs="device"),
        node(func=soc.get_ae_model, inputs=["features", "loaded_model"], outputs="model"),
        node(func=lod.get_mse_loss, inputs=None, outputs="loss"),
        node(func=lod.get_adam_optimizer, inputs=["model", "lr"], outputs="initialized_optimizer"),
        node(
            func=soc.rbm_init_ae,
            inputs=["model", "train_data_loader", "device", "is_model_initialized", "preprocessing"],
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
                "epoch": "loaded_epoch",
                "lr": "lr",
            },
            outputs="results",
        ),
    ]
)
