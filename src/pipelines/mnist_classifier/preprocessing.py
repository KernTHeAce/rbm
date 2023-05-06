from kedro.io import DataCatalog, MemoryDataSet
from kedro.pipeline import pipeline
from kedro.pipeline.node import node

from nodes.test_train import common
from nodes.test_train import loss_optim_device as lod
from nodes.test_train import mnist
from src import DATA_DIR, EXPERIMENTS_DIR
from src.common.const import SaverLoaderConst as slc
from src.nodes import output_concat
from src.nodes.save_load import load_state_dict

preprocessing_data = DataCatalog(
    {
        "mnist_train_dataset_path": MemoryDataSet(f"{DATA_DIR}/mnist/train"),
        "mnist_test_dataset_path": MemoryDataSet(f"{DATA_DIR}/mnist/test"),
        "train_data_loader": MemoryDataSet(copy_mode="assign"),
        "batch_size": MemoryDataSet(16),
        "shuffle": MemoryDataSet(True),
        "features": MemoryDataSet([28 * 28, 100, 50, 10]),
        "is_cuda": MemoryDataSet(False),
        "lr": MemoryDataSet(1e-3),
        "experiment_path": MemoryDataSet(f"{EXPERIMENTS_DIR}/test_2"),
        "checkpoint": MemoryDataSet(slc.LAST),
        "new_experiment": MemoryDataSet(True),
        "preprocessing": MemoryDataSet(lambda images: images.view(images.size()[0], -1)),
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
            func=mnist.get_mnist_dataset,
            inputs=["mnist_train_dataset_path"],
            outputs="mnist_train_dataset",
        ),
        node(
            func=mnist.get_mnist_dataset,
            inputs=["mnist_test_dataset_path"],
            outputs="mnist_test_dataset",
        ),
        node(
            func=common.dataset_to_dataloader,
            inputs=["mnist_train_dataset", "batch_size", "shuffle"],
            outputs="train_data_loader",
        ),
        node(
            func=common.dataset_to_dataloader,
            inputs=["mnist_test_dataset", "batch_size", "shuffle"],
            outputs="test_data_loader",
        ),
        node(func=lod.get_device, inputs="is_cuda", outputs="device"),
        node(func=mnist.get_classifier_model, inputs=["features", "loaded_model"], outputs="model"),
        node(func=lod.get_cross_entropy_loss, inputs=None, outputs="loss"),
        node(func=lod.get_adam_optimizer, inputs=["model", "lr"], outputs="initialized_optimizer"),
        node(
            func=mnist.rbm_init_classifier,
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
                "preprocessing": "preprocessing",
            },
            outputs="results",
        ),
    ]
)
