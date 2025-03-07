import torch
from core.models.classifier import Classifier
from copy import deepcopy
from torch import nn
from torch.optim import Adam
import datetime
from copy import deepcopy

import torch

from core.training import BaseTrainer, MlFlowLogger, model_training_pipeline
# from .utils import get_name_by_params
from custom.rbm.model_initializer.rbm_initializer import ModelRBMInitializer
from src import ADAM_EPOCHS, DEVICE, GRAD_MIN_MAX, LR
import torch
import torchvision
from torch.utils.data import DataLoader

from core.metrics import MetricCalculator, classification
from core.models import Classifier
from custom.rbm import generate_combinations, init_model_with_rbm_experiment
from src import BATCH_SIZE, DATA_DIR, DEVICE
from src.experiments import DEFAULT_RBM_EXPERIMENT_INIT_COMBINATIONS


class CumulativeRuleModel(nn.Module):
    def __init__(self, model):
        super(CumulativeRuleModel, self).__init__()
        self.forward_layers = nn.Sequential(*[item for item in model.model])
        self.reverse_layers = []
        for layer in reversed([item for item in model.model][:-1]):
            if isinstance(layer, nn.Linear):
                tmp = nn.Linear(in_features=layer.out_features, out_features=layer.in_features, dtype=torch.float64)
                # tmp.weight.data = torch.rand(layer.in_features, dtype=torch.float64)
                # tmp.bias.data = torch.rand(1, layer.in_features, dtype=torch.float64)
                layer = tmp
            self.reverse_layers.append(deepcopy(layer))
        self.reverse_layers.append(nn.ReLU())
        self.reverse_layers = nn.Sequential(*self.reverse_layers)

    def forward_step(self, data, model_layers, preprocess):
        results = []
        layers = []
        for layer in model_layers:
            layers.append(layer)
            if layer.__class__ in [nn.ReLU, nn.LeakyReLU, nn.Softmax]:
                tmp_model = nn.Sequential(*layers)
                data1 = preprocess(data)
                results.append(tmp_model(data1))
        return results

    def test(self, x_0):
        forward_0 = self.forward_step(x_0, self.forward_layers, lambda x: x)
        middle_0 = forward_0[-1]
        reverse_0 = self.forward_step(x_0, self.reverse_layers, nn.Sequential(*self.forward_layers))
        x_1 = reverse_0[-1]
        forward_1 = self.forward_step(x_1, self.forward_layers, lambda x: x)
        reverse_1 = self.forward_step(x_1, self.reverse_layers, nn.Sequential(*self.forward_layers))
        return forward_0, reverse_0, middle_0, forward_1, reverse_1


class CustomLoss(nn.Module):
    def __init__(self, model_loss):
        super(CustomLoss, self).__init__()
        self.mse = torch.nn.MSELoss()
        self.loss = model_loss

    def forward(self, y_forward_0, y_reverse_0, y_middle, y_forward_1, y_reverse_1, target, model):
        losses = []
        for i, item in enumerate(y_forward_0):
            a = torch.autograd.grad(self.mse(y_forward_1[i], item) + self.loss(y_middle, target), model.forward_layers[0].weight)[0]
            b = model.forward_layers[0].weight
            losses.append(self.mse(y_forward_1[i], item) + self.loss(y_middle, target))
        for i, item in enumerate(y_reverse_0):
            losses.append(self.mse(y_reverse_1[i], item))
        return torch.cat([losses[i].reshape(1) for i in range(len(losses))])


class CumulativeRuleInitializer:
    def __init__(self, trainer, device, lr, epochs=None):
        self.trainer = trainer
        self.epochs = epochs
        self.device = device
        self.lr = lr

    def update_params(self, model: CumulativeRuleModel, y_forward_0, y_reverse_0, y_middle, y_forward_1, y_reverse_1, target, loss):
        i = 0
        mse = torch.nn.MSELoss()
        for layer in model.forward_layers:
            if isinstance(layer, nn.Linear):
                error = mse(y_forward_1[i], y_forward_0[i]) + loss(y_middle, target)
                grad_w = torch.autograd.grad(error, layer.weight)[0]
                grad_b = torch.autograd.grad(error, layer.bias)[0]
                with torch.no_grad:
                    layer.weight -= LR * grad_w
                    layer.bias -= LR * grad_b
                i += 1

        i = 0
        for layer in model.reverse_layers:
            if isinstance(layer, nn.Linear):
                error = mse(y_reverse_1[i], y_reverse_0[i])
                layer.weight -= LR * torch.autograd.grad(error, layer.weight)[0]
                layer.bias -= LR * torch.autograd.grad(error, layer.bias)[0]
                i += 1
        return model

    def __call__(self, model):
        model = CumulativeRuleModel(model)
        for epoch in range(self.epochs):
            for batch in self.trainer.train_loader:
                input_, target = self.trainer.get_data(batch)
                y_forward_0, y_reverse_0, y_middle, y_forward_1, y_reverse_1 = model.test(input_)
                model = self.update_params(model, y_forward_0, y_reverse_0, y_middle, y_forward_1, y_reverse_1, target, self.trainer.loss)
        return model

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

model = Classifier([784, 100, 10])
trainer = BaseTrainer(
        torch.optim.Adam,
        LR,
        torch.nn.CrossEntropyLoss(),
        DEVICE,
        train_loader=train_loader,
        test_loader=test_loader,
        preprocessing=lambda img: img.view(-1, 28 * 28),
        postprocessing=lambda outputs: torch.tensor([torch.argmax(batch).item() for batch in outputs]).to(DEVICE),
    )
initializer = CumulativeRuleInitializer(trainer, 1, 1, 1)
model = initializer(model)
