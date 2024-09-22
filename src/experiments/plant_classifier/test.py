import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from core import BaseTrainer, MetricCalculator, MlFlowLogger, metrics, model_training_pipeline
from src import ADAM_EPOCHS, DEVICE
from src.experiments.plant_classifier.model import PlantClassifier

data_transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
model = PlantClassifier()
model.load_state_dict(
    torch.load("/home/kern/PycharmProjects/rbm/src/experiments/plant_classifier/60/model", weights_only=False)
)
orig_set = ImageFolder("/home/kern/PycharmProjects/rbm/data/plants_test", transform=data_transform)
test_loader = DataLoader(orig_set, batch_size=3, shuffle=False)
for batch in test_loader:
    print(torch.tensor([torch.argmax(item).item() for item in model(batch[0])]).to(DEVICE))
