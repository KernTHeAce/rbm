import torch
import torchvision
from torch.utils.data import DataLoader

from core.metrics import MetricCalculator, regression
from core.models import BaseModel
from custom.rbm import generate_combinations, run_experiment
from src import BATCH_SIZE, DATA_DIR, DEVICE
from src.experiments import INITIALIZERS

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(
        DATA_DIR,
        train=True,
        download=True,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5,), (0.5,)),
                torchvision.transforms.ConvertImageDtype(torch.double),
            ]
        ),
    ),
    batch_size=BATCH_SIZE,
    shuffle=True,
)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(
        DATA_DIR,
        train=False,
        download=True,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5,), (0.5,)),
                torchvision.transforms.ConvertImageDtype(torch.double),
            ]
        ),
    ),
    batch_size=BATCH_SIZE,
    shuffle=True,
)

lengths = {"s": [35], "m": [50, 20, 50], "l": [75, 50, 35, 20, 35, 50, 75]}

model_combinations = generate_combinations(
    {
        "l": ["s", "m", "l"],
        "w_k": [1, 7, 15],
    }
)
MODEL_INPUT_SIZE = 28 * 28
metrics_calculator = MetricCalculator([regression.mae()])

for model_params in model_combinations:
    model = BaseModel(
        [MODEL_INPUT_SIZE]
        + [item * model_params["w_k"] for item in lengths[model_params["l"]]]
        + [MODEL_INPUT_SIZE]
    ).to(DEVICE)
    for initializer_type, value in INITIALIZERS.items():
        for initializer_params in value:
            response = run_experiment(
                test_loader,
                train_loader,
                f"mnist_l={model_params['l']}_wk={model_params['w_k']}",
                model,
                torch.nn.MSELoss(),
                initializer_params,
                initializer_type,
                metrics_calculator,
                lambda img: img.view(-1, 28 * 28),
                lambda outputs: outputs,
            )
