from common import Model
from common.experiment import rbm_experiment, generate_combinations
from dataset import SoccerCSVDataSet

from src import DATA_DIR, DEVICE, BATCH_SIZE
from src import DEVICE, ADAPTIVE_LRS, ADAM_EPOCHS, INITIALIZER_EPOCHS, GRAD_MIN_MAX

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

reference_combinations = [{"adaptive_lr": False}]

all_combinations = reference_combinations + rbm_combinations + rbm_adaptive_combinations
AUTOENCODER_INPUT_SIZE = 39

for w_k in [1, 7, 15]:
    for key, value in lengths.items():
        model = Model([AUTOENCODER_INPUT_SIZE] + [item * w_k for item in value] + [AUTOENCODER_INPUT_SIZE]).to(DEVICE)
        rbm_experiment(
            test_loader=test_loader,
            train_loader=train_loader,
            experiment_name=f"1_soccer_l={key}_wk={w_k}",
            model=model
        )
