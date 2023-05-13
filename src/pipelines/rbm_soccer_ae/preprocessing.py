from kedro.pipeline import pipeline
from kedro.pipeline.node import node

from nodes.test_train import common
from nodes.test_train import loss_optim_device as lod
from nodes.test_train import soccer as soc
from src.nodes import output_concat
from src.nodes.save_load import load_state_dict

preprocessing_pipeline = pipeline(
    [
        node(
            func=load_state_dict,
            inputs=["experiment_name", "checkpoint", "new_experiment"],
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
        node(
            func=soc.rbm_init_ae,
            inputs=["model", "train_data_loader", "device", "is_model_initialized", "preprocessing"],
            outputs="initialized_model",
        ),
        node(func=lod.get_adam_optimizer, inputs=["model", "lr", "loaded_optimizer"], outputs="initialized_optimizer"),
        node(
            func=output_concat,
            inputs={
                "initialized_model": "model",
                "initialized_optimizer": "initialized_optimizer",
                "device": "device",
                "train_data_loader": "train_data_loader",
                "test_data_loader": "test_data_loader",
                "loss": "loss",
                "epoch": "loaded_epoch",
                "lr": "lr",
                "preprocessing": "preprocessing",
                "experiment_name": "experiment_name",
            },
            outputs="results",
        ),
    ]
)
