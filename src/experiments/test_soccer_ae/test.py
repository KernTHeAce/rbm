from common import Model
from common.experiment import rbm_experiment
from dataset import SoccerCSVDataSet

from src import DATA_DIR, DEVICE, BATCH_SIZE

from torch.utils.data import DataLoader
from common import BaseTrainer, pipeline, MetricCalculator, metrics, MlFlowLogger
from common.model_initializer.initializer import ModelRBMInitializer
from copy import deepcopy
import torch
import datetime

from src import DEVICE, ADAPTIVE_LRS, ADAM_EPOCHS, INITIALIZER_EPOCHS







# Пример использования
parameters = {"param_1": [1, 2], "param_2": [0, 1, 2]}
combinations = generate_combinations(parameters)

for combination in combinations:
    print(combination)
#
# train_set = SoccerCSVDataSet(f"{DATA_DIR}/soccer/01_raw/wiscout/train_x_sigm_1221.csv")
# train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
# test_set = SoccerCSVDataSet(f"{DATA_DIR}/soccer/01_raw/wiscout/test_x_sigm_136.csv")
# test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)
# lengths = {
#     "s": [16],
#     "m": [29, 16, 29],
#     "l": [34, 29, 24, 16, 24, 29, 34]
# }
#
# AUTOENCODER_INPUT_SIZE = 39
#
# model = Model([AUTOENCODER_INPUT_SIZE] + [29, 16, 29] + [AUTOENCODER_INPUT_SIZE]).to(DEVICE)
# trainer = BaseTrainer(
#     torch.optim.Adam,
#     1e-3,
#     torch.nn.MSELoss(),
#     DEVICE,
#     train_loader=train_loader,
#     test_loader=test_loader,
# )
#
# model = Model([AUTOENCODER_INPUT_SIZE] + [34, 29, 24, 16, 24, 29, 34] + [AUTOENCODER_INPUT_SIZE]).to(DEVICE)
# rbm_experiment(
#     test_loader=test_loader,
#     train_loader=train_loader,
#     experiment_name=f"Default",
#     model=model
# )
