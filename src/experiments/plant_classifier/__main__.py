from pathlib import Path

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from core import BaseTrainer, MetricCalculator, MlFlowLogger, metrics, model_training_pipeline
from src import ADAM_EPOCHS, DATA_DIR, DEVICE
from src.experiments.plant_classifier.model import PlantClassifier

data_transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])

NUM_CLASSES = 14

orig_set = ImageFolder(str(Path(DATA_DIR, "custom_plants_dataset_balanced")), transform=data_transform)  # your dataset
train_set_size = int(len(orig_set) * 0.9)
train_set, test_set = torch.utils.data.random_split(orig_set, [train_set_size, len(orig_set) - train_set_size])

test_loader = DataLoader(test_set, batch_size=32, shuffle=True)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)

model = PlantClassifier().to(DEVICE)

trainer = BaseTrainer(
    torch.optim.Adam,
    1e-3,
    torch.nn.CrossEntropyLoss(),
    DEVICE,
    train_loader=train_loader,
    test_loader=test_loader,
    postprocessing=lambda outputs: torch.tensor([torch.argmax(batch).item() for batch in outputs]).to(DEVICE),
)
logger = MlFlowLogger(f"plants_balanced", "60")
metrics_calculator = MetricCalculator(
    [
        metrics.f1(num_classes=NUM_CLASSES),
        metrics.accuracy(num_classes=NUM_CLASSES),
        metrics.recall(num_classes=NUM_CLASSES),
    ]
)
model, optimizer = model_training_pipeline(model, trainer, 60, metrics_calculator, logger, model_initializer=None)

torch.save(model.state_dict(), "/home/kern/PycharmProjects/rbm/src/experiments/plant_classifier/model")
torch.save(optimizer.state_dict(), "/home/kern/PycharmProjects/rbm/src/experiments/plant_classifier/optimizer")
