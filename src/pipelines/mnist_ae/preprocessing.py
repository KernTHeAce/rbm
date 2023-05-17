from kedro.pipeline import pipeline
from kedro.pipeline.node import node

from src.nodes import output_concat
from src.nodes.save_load import load_state_dict
from src.nodes.test_train import common
from src.nodes.test_train import loss_optim_device as lod
from src.nodes.test_train import mnist
from src.nodes.test_train import soccer as soc

preprocessing_pipeline = pipeline(
    [
        node(
            func=load_state_dict,
            inputs=["experiment_name", "checkpoint", "new_experiment"],
            outputs=["loaded_model", "loaded_optimizer", "loaded_epoch", "is_model_initialized"],
        ),
        # node(
        #     func=mnist.get_mnist_dataset,
        #     inputs=["mnist_train_dataset_path"],
        #     outputs="mnist_train_dataset",
        # ),
        # node(
        #     func=mnist.get_mnist_dataset,
        #     inputs=["mnist_test_dataset_path"],
        #     outputs="mnist_test_dataset",
        # ),
        node(
            func=mnist.get_small_mnist_datasets,
            inputs=["mnist_test_dataset_path"],
            outputs=["mnist_train_dataset", "mnist_test_dataset"],
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
        node(func=soc.get_ae_model, inputs=["features", "loaded_model"], outputs="model"),
        node(func=lod.get_mse_loss, inputs=None, outputs="loss"),
        node(func=lod.get_adam_optimizer, inputs=["model", "lr"], outputs="initialized_optimizer"),
        node(
            func=soc.rbm_init_ae,
            inputs=[
                "model",
                "train_data_loader",
                "device",
                "is_model_initialized",
                "preprocessing",
                "rbm_epoch",
                "rbm_type",
                "rbm_init_type",
            ],
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
                "experiment_name": "experiment_name",
                "rbm_epoch": "rbm_epoch",
                "rbm_init_type": "rbm_init_type",
                "rbm_type": "rbm_type",
            },
            outputs="results",
        ),
    ]
)
