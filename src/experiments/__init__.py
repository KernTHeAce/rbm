from custom.rbm import generate_combinations
from src import INITIALIZER_EPOCHS
INITIALIZERS = {
    "reference": [{}],
    "rbm": generate_combinations({
        "epochs": INITIALIZER_EPOCHS,
        "adaptive_lr": [True, False],
        "semisupervised_learning": [True, False]
    }),
    "cr": generate_combinations({
        "epochs": INITIALIZER_EPOCHS,
    })

}
