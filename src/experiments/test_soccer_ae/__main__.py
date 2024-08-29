from common import Model
from common.experiment import rbm_experiment, generate_combinations
from dataset import SoccerCSVDataSet

from src import DATA_DIR, BATCH_SIZE, INITIALIZER_EPOCHS, DEVICE

from torch.utils.data import DataLoader

model = Model([39, 34, 29, 24, 16, 24, 29, 34, 39]).to(DEVICE)
train_set = SoccerCSVDataSet(f"{DATA_DIR}/soccer/01_raw/wiscout/train_x_sigm_1221.csv")
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_set = SoccerCSVDataSet(f"{DATA_DIR}/soccer/01_raw/wiscout/test_x_sigm_136.csv")
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)
lengths = {
    "s": [16],
    "m": [29, 16, 29],
    "l": [34, 29, 24, 16, 24, 29, 34]
}

rbm_adaptive_combinations = generate_combinations({
    "grad_clipping": [True, False],
    "epochs": INITIALIZER_EPOCHS,
    "adaptive_lr": [True]
})

rbm_combinations = generate_combinations({
    "epochs": INITIALIZER_EPOCHS,
    "adaptive_lr": [False]
})

reference_combinations = [{"adaptive_lr": None}]

initializer_combinations = reference_combinations + rbm_combinations + rbm_adaptive_combinations
model_combinations = generate_combinations({
    "l": ["s", "m", "l"],
    "w_k": [1, 7, 15],
})
AUTOENCODER_INPUT_SIZE = 39

for model_params in model_combinations:
    model = Model(
        [AUTOENCODER_INPUT_SIZE] +
        [item * model_params["w_k"] for item in lengths[model_params["l"]]] +
        [AUTOENCODER_INPUT_SIZE]
    ).to(DEVICE)

    rbm_experiment(
        test_loader=test_loader,
        train_loader=test_loader,
        experiment_name=f"soccer_l={model_params['l']}_wk={model_params['w_k']}",
        model=model,
        params=initializer_combinations
    )
