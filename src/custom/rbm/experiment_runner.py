import datetime
from copy import deepcopy

import torch

from core.training import BaseTrainer, MlFlowLogger, model_training_pipeline
from .utils import get_name_by_params
from custom.rbm.model_initializer.initializer import ModelRBMInitializer
from src import ADAM_EPOCHS, DEVICE, GRAD_MIN_MAX


def init_model_with_rbm_experiment(
    test_loader,
    train_loader,
    experiment_name,
    model,
    loss,
    params,
    metrics_calculator,
    preprocessing=lambda x: x,
    postprocessing=lambda x: x,
):
    for param in params:
        trainer = BaseTrainer(
            torch.optim.Adam,
            1e-3,
            loss,
            DEVICE,
            train_loader=train_loader,
            test_loader=test_loader,
            preprocessing=preprocessing,
            postprocessing=postprocessing,
        )
        model_initializer = ModelRBMInitializer(trainer, device=DEVICE, lr=1e-3, grad_min_max=GRAD_MIN_MAX, **param)
        run_name = get_name_by_params(param)
        logger = MlFlowLogger(experiment_name, run_name)
        print(f"{datetime.datetime.now().strftime('%H:%M:%S.%f')[:-7]}  -  start {experiment_name} {run_name}")
        model_training_pipeline(
            deepcopy(model), trainer, ADAM_EPOCHS, metrics_calculator, logger, model_initializer=model_initializer
        )
        print(f"{datetime.datetime.now().strftime('%H:%M:%S.%f')[:-7]}  -  finish {experiment_name} {run_name}\n")
