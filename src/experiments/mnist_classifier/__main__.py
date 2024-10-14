import torch
import torchvision
from torch.utils.data import DataLoader

from core.metrics import MetricCalculator, classification
from core.models import Classifier
from custom.rbm import generate_combinations, init_model_with_rbm_experiment
from src import BATCH_SIZE, DATA_DIR, DEVICE
from src.experiments import DEFAULT_RBM_EXPERIMENT_INIT_COMBINATIONS

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

lengths = {"s": [15], "m": [35, 15, 10], "l": [50, 35, 20, 15, 10]}
model_combinations = generate_combinations(
    {
        # "l": ["s", "m", "l"],
        "l": ["l"],
        "w_k": [1, 7, 15],
    }
)
MODEL_INPUT_SIZE = 28 * 28
MODEL_OUTPUT_SIZE = 10
metrics_calculator = MetricCalculator([classification.f1(num_classes=10)])

for model_params in model_combinations:
    model = Classifier(
        [MODEL_INPUT_SIZE] + [item * model_params["w_k"] for item in lengths[model_params["l"]]] + [MODEL_OUTPUT_SIZE]
    ).to(DEVICE)
    init_model_with_rbm_experiment(
        test_loader=test_loader,
        train_loader=train_loader,
        experiment_name=f"12mnist_l={model_params['l']}_wk={model_params['w_k']}",
        model=model,
        loss=torch.nn.CrossEntropyLoss(),
        params=DEFAULT_RBM_EXPERIMENT_INIT_COMBINATIONS[:],
        metrics_calculator=metrics_calculator,
        preprocessing=lambda img: img.view(-1, 28 * 28),
        postprocessing=lambda outputs: torch.tensor([torch.argmax(batch).item() for batch in outputs]).to(DEVICE),
    )
