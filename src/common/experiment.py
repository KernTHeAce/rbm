from common import BaseTrainer, pipeline, MetricCalculator, metrics, MlFlowLogger
from common.model_initializer.initializer import ModelRBMInitializer
from copy import deepcopy

from src import DEVICE, ADAM_EPOCHS, GRAD_MIN_MAX

import torch
import datetime
from itertools import product


def generate_combinations(parameters):
    keys = list(parameters.keys())
    values = list(parameters.values())
    combinations = [dict(zip(keys, values_tuple)) for values_tuple in product(*values)]
    return combinations


def get_name_by_params(params):
    if params["adaptive_lr"] is None:
        return "reference"
    if params["adaptive_lr"]:
        return f"rbm_adapt_{params['epochs']}_{params['grad_clipping']}"
    return f"rbm_adapt_{params['epochs']}"


def rbm_experiment(test_loader, train_loader, experiment_name, model, params):
    metrics_calculator = MetricCalculator([metrics.mse])
    for param in params:
        model_initializer = ModelRBMInitializer(
            test_loader,
            device=DEVICE,
            lr=1e-3,
            grad_min_max=GRAD_MIN_MAX,
            **param
        )
        trainer = BaseTrainer(
            torch.optim.Adam,
            1e-3,
            torch.nn.MSELoss(),
            DEVICE,
            train_loader=train_loader,
            test_loader=test_loader,
        )
        run_name = get_name_by_params(param)
        logger = MlFlowLogger(experiment_name, run_name)
        print(f"{datetime.datetime.now().strftime('%H:%M:%S.%f')[:-7]}  -  start {experiment_name} {run_name}")
        pipeline(
            deepcopy(model), trainer, ADAM_EPOCHS, metrics_calculator, logger, model_initializer=model_initializer
        )
        print(f"{datetime.datetime.now().strftime('%H:%M:%S.%f')[:-7]}  -  finish {experiment_name} {run_name}")
