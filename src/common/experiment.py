from common import BaseTrainer, pipeline, MetricCalculator, metrics, MlFlowLogger
from common.model_initializer.initializer import ModelRBMInitializer
from copy import deepcopy

from src import DEVICE, ADAPTIVE_LRS, ADAM_EPOCHS, INITIALIZER_EPOCHS, GRAD_MIN_MAX

import torch
import datetime
from itertools import product


RUN_NAME_PREFIX = {
    True: "rbm_adaptive",
    False: "rbm",
    None: "classic"
}

def generate_combinations(parameters):
    keys = list(parameters.keys())
    values = list(parameters.values())

    combinations = [dict(zip(keys, values_tuple)) for values_tuple in product(*values)]

    return combinations


def rbm_experiment(test_loader, train_loader, experiment_name, model):
    trainer = BaseTrainer(
        torch.optim.Adam,
        1e-3,
        torch.nn.MSELoss(),
        DEVICE,
        train_loader=train_loader,
        test_loader=test_loader,
    )

    metrics_calculator = MetricCalculator([metrics.mse])
    for adaptive_lr in ADAPTIVE_LRS:
        for epochs in INITIALIZER_EPOCHS:
            for grad_clipping in [True, False]:
                if not adaptive_lr and adaptive_lr:
                    continue
                model_initializer = ModelRBMInitializer(
                    test_loader,
                    epochs,
                    DEVICE,
                    lr=1e-3,
                    grad_min_max=GRAD_MIN_MAX,
                    use_grad_clipping=grad_clipping,
                    adaptive_lr=adaptive_lr,
                )
                run_name = f"{RUN_NAME_PREFIX[adaptive_lr]}_{epochs}" \
                    if adaptive_lr is not None else RUN_NAME_PREFIX[adaptive_lr]
                if adaptive_lr:
                    run_name += f"_gc={grad_clipping}"
                logger = MlFlowLogger(experiment_name, run_name)
                print(f"{datetime.datetime.now().strftime('%H:%M:%S.%f')[:-7]}  -  start {experiment_name} {run_name}")
                pipeline(
                    deepcopy(model), trainer, ADAM_EPOCHS, metrics_calculator, logger, model_initializer=model_initializer
                )
                print(f"{datetime.datetime.now().strftime('%H:%M:%S.%f')[:-7]}  -  finish {experiment_name} {run_name}")

            if adaptive_lr is None:
                break
