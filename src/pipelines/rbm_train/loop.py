from kedro.runner import SequentialRunner
from kedro.io import DataCatalog, MemoryDataSet
from kedro.pipeline import pipeline
from kedro.extras.datasets.pandas import CSVDataSet
from kedro.pipeline.node import node
from src import BASE_DIR
from src.nodes.test_train import (
    get_device,
    csv_to_data_loader,
    get_ae_model,
    get_loss,
    get_optimizer,
    rbm_init_ae,
)
from src.nodes import output_concat
from src.nodes.save_load import load_state_dict
from src.common.const import CommonConst as cc
from src.common.const import SaverLoaderConst as slc
from src.common.utils import update_datacatalog
from src.pipelines.rbm_train.epoch import epoch_data, epoch_pipeline
MAX_EPOCH = 200
runner = SequentialRunner()

preprocessing_data = DataCatalog({
    "soccer_train_dataset": CSVDataSet(filepath=f"{BASE_DIR}/data/01_raw/wiscout/train_x_sigm_1221.csv"),
    "soccer_test_dataset": CSVDataSet(filepath=f"{BASE_DIR}/data/01_raw/wiscout/test_x_sigm_136.csv"),
    "train_data_loader": MemoryDataSet(copy_mode="assign"),
    "batch_size": MemoryDataSet(16),
    "shuffle": MemoryDataSet(True),
    "features": MemoryDataSet([39, 34, 29, 24, 19]),
    "is_cuda": MemoryDataSet(True),
    "lr": MemoryDataSet(1e-3),
    "experiment_path": MemoryDataSet(f"{BASE_DIR}/experiments/test_1"),
    "checkpoint": MemoryDataSet(slc.LAST),
    "new_experiment": MemoryDataSet(True),
})

preprocessing_pipeline = pipeline([
        node(func=load_state_dict,
             inputs=["experiment_path", "checkpoint", "new_experiment"],
             outputs=["loaded_model", "loaded_optimizer", "is_model_initialized"]),
        node(func=csv_to_data_loader,
             inputs=["soccer_train_dataset", "batch_size", "shuffle"],
             outputs="train_data_loader"),
        node(func=csv_to_data_loader,
             inputs=["soccer_test_dataset", "batch_size", "shuffle"],
             outputs="test_data_loader"),
        node(func=get_device,
             inputs="is_cuda",
             outputs="device"),
        node(func=get_ae_model,
             inputs=["features", "loaded_model"],
             outputs="model"),
        node(func=get_loss,
             inputs=None,
             outputs="loss"),
        node(func=get_optimizer,
             inputs=["model", "lr"],
             outputs="initialized_optimizer"),
        node(func=rbm_init_ae,
             inputs=["model", "train_data_loader", "device", "is_model_initialized"],
             outputs="initialized_model"),
        node(func=output_concat,
             inputs={
                "initialized_model": "initialized_model",
                "initialized_optimizer": "initialized_optimizer",
                "device": "device",
                "train_data_loader": "train_data_loader",
                "test_data_loader": "test_data_loader",
                "loss": "loss"
             },
             outputs="results"),
])

preprocessing_output = runner.run(preprocessing_pipeline, preprocessing_data)
print(preprocessing_output.keys())
loop_data = update_datacatalog(epoch_data, preprocessing_output["results"])
for i in range(MAX_EPOCH):
    print(i)
    epoch_output = runner.run(epoch_pipeline, loop_data)
    loop_data = update_datacatalog(loop_data, epoch_output["results"], replace=True)

