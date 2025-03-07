from custom.rbm import generate_combinations
from src import INITIALIZER_EPOCHS

rbm_combinations = generate_combinations({
    "epochs": INITIALIZER_EPOCHS,
    "adaptive_lr": [False, True],
    "semisupervised_learning": [True, False]
})

reference_combinations = [{"adaptive_lr": None}]

DEFAULT_RBM_EXPERIMENT_INIT_COMBINATIONS = rbm_combinations + reference_combinations
