import datetime
from copy import deepcopy

import torch

from core.training import BaseTrainer, MlFlowLogger, model_training_pipeline
from .utils import get_name_by_params
from custom.rbm.model_initializer.initializer import ModelRBMInitializer
from src import ADAM_EPOCHS, DEVICE, GRAD_MIN_MAX


def run_experiment(
    test_loader,
    train_loader,
    experiment_name,
    model,
    loss,
    param,
    metrics_calculator,
    preprocessing=lambda x: x,
    postprocessing=lambda x: x,
):
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
    response = model_training_pipeline(
        deepcopy(model), trainer, ADAM_EPOCHS, metrics_calculator, logger, model_initializer=model_initializer
    )
    print(f"{datetime.datetime.now().strftime('%H:%M:%S.%f')[:-7]}  -  finish {experiment_name} {run_name}\n")
    return response


def remove_useless_params(current_param, params):
    grad_clipping = current_param.get("grad_clipping")
    epochs = current_param.get("epochs")
    adaptive_lr = current_param.get("adaptive_lr")
    params_copy = params[:]
    for index_, param in enumerate(params_copy):
        if param.get("grad_clipping") == grad_clipping and param.get("adaptive_lr") == adaptive_lr and (x := param.get("epochs")) is not None and x > epochs:
            params.pop(index_)
    return params


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
    while len(params):
        current_param = params.pop(0)
        response = run_experiment(
            test_loader, train_loader, experiment_name, model, loss, current_param, metrics_calculator, preprocessing, postprocessing,
        )
        if response is None:
            updated_params = remove_useless_params(current_param, params)
            init_model_with_rbm_experiment(
                test_loader, train_loader, experiment_name, model, loss, updated_params, metrics_calculator, preprocessing, postprocessing,
            )
            return
