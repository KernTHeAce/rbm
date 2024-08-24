from common import BaseTrainer, pipeline, Model, MetricCalculator, metrics, MlFlowLogger
from common.model_initializer.initializer import ModelRBMInitializer
from dataset import SoccerCSVDataSet

from src import DATA_DIR, DEVICE

import torch
from torch.utils.data import DataLoader

BATCH_SIZE = 8
SHUFFLE = True
EPOCHS = 100

model = Model([39, 34, 29, 24, 16, 24, 29, 34, 39]).to(DEVICE)
train_set = SoccerCSVDataSet(f"{DATA_DIR}/soccer/01_raw/wiscout/train_x_sigm_1221.csv")
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=SHUFFLE)
test_set = SoccerCSVDataSet(f"{DATA_DIR}/soccer/01_raw/wiscout/test_x_sigm_136.csv")
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=SHUFFLE)

model_initializer = ModelRBMInitializer(
    train_loader, 3, DEVICE
)

trainer = BaseTrainer(
    torch.optim.Adam,
    1e-3,
    torch.nn.MSELoss(),
    DEVICE,
    train_loader=train_loader,
    test_loader=test_loader,
)

logger = MlFlowLogger("My_experiment", "no_rbm")

metrics_calculator = MetricCalculator([metrics.mse])

pipeline(model, trainer, EPOCHS, metrics_calculator, logger, model_initializer=False)
