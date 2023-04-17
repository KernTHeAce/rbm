from src.nodes.test_train import (
    get_device,
    one_epoch_ae_train,
    csv_to_data_loader,
    get_ae_model,
    get_loss,
    get_optimizer,
    rbm_init_ae,
    test,
)
from src import BASE_DIR
from kedro.pipeline.node import node
from kedro.pipeline import pipeline
from kedro.runner import SequentialRunner
from kedro.io import DataCatalog, MemoryDataSet
from kedro.extras.datasets.pandas import CSVDataSet
from src.nodes.metrics import mse_metric
from src.nodes.save_load import save_state_dict, load_state_dict


data_catalog = DataCatalog({
    "soccer_train_dataset": CSVDataSet(filepath=f"{BASE_DIR}/data/01_raw/wiscout/train_x_sigm_1221.csv"),
    "soccer_test_dataset": CSVDataSet(filepath=f"{BASE_DIR}/data/01_raw/wiscout/test_x_sigm_136.csv"),
    "train_data_loader": MemoryDataSet(copy_mode="assign"),
    "batch_size": MemoryDataSet(16),
    "shuffle": MemoryDataSet(True),
    "features": MemoryDataSet([39, 34, 29, 24, 19]),
    "is_cuda": MemoryDataSet(True),
    "lr": MemoryDataSet(1e-3),
    "max_epoch": MemoryDataSet(200),
    "data_preprocess": MemoryDataSet(lambda x: x),

    "experiment_path": MemoryDataSet(f"{BASE_DIR}/experiments/test_1"),
})
print(BASE_DIR)

soccer_data_process = pipeline([
        node(func=load_state_dict,
             inputs=["experiment_path"],
             outputs=["loaded_model", "loaded_optimizer"]),
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
             inputs=["features"],
             outputs="model"),
        node(func=get_loss,
             inputs=None,
             outputs="loss"),
        node(func=get_optimizer,
             inputs=["model", "lr"],
             outputs="optimizer"),
        node(
          func=rbm_init_ae,
          inputs=["model", "train_data_loader", "device"],
          outputs="initialized_model"
        ),
        node(func=one_epoch_ae_train,
             inputs=["initialized_model", "optimizer", "loss", "train_data_loader", "device", "data_preprocess"],
             outputs=["train_time", "train_average_loss", "updated_optimizer"]),

        node(func=test,
             inputs=["initialized_model", "loss", "test_data_loader", "device", "data_preprocess"],
             outputs=["test_time", "test_average_loss", "y_true", "y_pred"]),
        node(func=mse_metric,
             inputs=["y_true", "y_pred"],
             outputs="metrics"),
        node(func=save_state_dict,
             inputs=["experiment_path", "metrics", "model", "optimizer"],
             outputs="output"),
        #
        # node(func=load_from_checkpoint,
        #      inputs=["experiment_path"],
        #      outputs="out")

    ])
runner = SequentialRunner()
output = runner.run(soccer_data_process, data_catalog)
print(output)
