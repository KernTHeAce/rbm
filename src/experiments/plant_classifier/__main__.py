from pathlib import Path

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision.datasets import ImageFolder
from core.training import BaseTrainer, MlFlowLogger, model_training_pipeline

from core.metrics import MetricCalculator, classification
from src import DATA_DIR, DEVICE
from src.experiments.plant_classifier.model import PlantClassifier

from torchvision import models


NUM_CLASSES = 14


def get_resnet_model(constructor):
    # Initialize pre-trained model
    pretrained_model = constructor(pretrained=True)

    pretrained_model.fc = torch.nn.Linear(pretrained_model.fc.in_features, NUM_CLASSES)

    # Freeze the parameters of the pre-trained layers
    for param in pretrained_model.parameters():
        param.requires_grad = False

    # Unfreeze the parameters of the last few layers for fine-tuning
    for param in pretrained_model.layer4.parameters():
        param.requires_grad = True
    return pretrained_model.to(DEVICE)


data_transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])

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


metrics_calculator = MetricCalculator(
    [
        classification.f1(num_classes=NUM_CLASSES),
        # classification.accuracy(num_classes=NUM_CLASSES),
        # classification.recall(num_classes=NUM_CLASSES),
    ]
)

pretrained_models = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
    "resnet101": models.resnet101,
    "resnet152": models.resnet152,
}
for key, item in pretrained_models.items():
    logger = MlFlowLogger(f"plants_balanced", key)

    model, optimizer = model_training_pipeline(get_resnet_model(item), trainer, 30, metrics_calculator, logger, model_initializer=None)

# torch.save(model.state_dict(), "/home/kern/PycharmProjects/rbm/src/experiments/plant_classifier/model")
# torch.save(optimizer.state_dict(), "/home/kern/PycharmProjects/rbm/src/experiments/plant_classifier/optimizer")
