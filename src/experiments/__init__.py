from custom.rbm import generate_combinations
from src import INITIALIZER_EPOCHS

rbm_adaptive_combinations = generate_combinations(
    {"grad_clipping": [True, False], "epochs": INITIALIZER_EPOCHS, "adaptive_lr": [True]}
)

rbm_combinations = generate_combinations({"epochs": INITIALIZER_EPOCHS, "adaptive_lr": [False]})

reference_combinations = [{"adaptive_lr": None}]

DEFAULT_RBM_EXPERIMENT_INIT_COMBINATIONS = reference_combinations + rbm_combinations + rbm_adaptive_combinations
