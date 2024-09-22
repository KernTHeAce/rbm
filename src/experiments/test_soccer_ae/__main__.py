import torch
from dataset import SoccerCSVDataSet
from torch.utils.data import DataLoader

from core.metrics import MetricCalculator, regression
from core.models import BaseModel
from custom.rbm import generate_combinations, init_model_with_rbm_experiment
from src import BATCH_SIZE, DATA_DIR, DEVICE, INITIALIZER_EPOCHS

train_loader = DataLoader(
    SoccerCSVDataSet(f"{DATA_DIR}/soccer/01_raw/wiscout/train_x_sigm_1221.csv"), batch_size=BATCH_SIZE, shuffle=True
)
test_loader = DataLoader(
    SoccerCSVDataSet(f"{DATA_DIR}/soccer/01_raw/wiscout/test_x_sigm_136.csv"), batch_size=BATCH_SIZE, shuffle=True
)

lengths = {"s": [16], "m": [29, 16, 29], "l": [34, 29, 24, 16, 24, 29, 34]}

rbm_adaptive_combinations = generate_combinations(
    {"grad_clipping": [True, False], "epochs": INITIALIZER_EPOCHS, "adaptive_lr": [True]}
)

rbm_combinations = generate_combinations({"epochs": INITIALIZER_EPOCHS, "adaptive_lr": [False]})

reference_combinations = [{"adaptive_lr": None}]

initializer_combinations = reference_combinations + rbm_combinations + rbm_adaptive_combinations
model_combinations = generate_combinations(
    {
        "l": ["s", "m", "l"],
        "w_k": [1, 7, 15],
    }
)
AUTOENCODER_INPUT_SIZE = 39

metrics_calculator = MetricCalculator([regression.mse])

for model_params in model_combinations:
    model = BaseModel(
        [AUTOENCODER_INPUT_SIZE]
        + [item * model_params["w_k"] for item in lengths[model_params["l"]]]
        + [AUTOENCODER_INPUT_SIZE]
    ).to(DEVICE)

    init_model_with_rbm_experiment(
        test_loader=test_loader,
        train_loader=train_loader,
        experiment_name=f"soccer_l={model_params['l']}_wk={model_params['w_k']}",
        model=model,
        loss=torch.nn.MSELoss(),
        params=initializer_combinations,
        metrics_calculator=metrics_calculator,
    )
