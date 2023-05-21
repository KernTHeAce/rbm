from collections import OrderedDict
from time import time
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms

from src.common.const import CommonConst as cc
from src.common.const import MetricConst as mc
from src.common.const import MetricsOutputValues as mov
from src.common.const import RBMInitTypes as rit
from src.common.const import RBMTypes as rt
from src.common.utils.average import Average
from src.models.mnist_classifier.small import Classifier
from src.models.rbm.manual_linear_rbm_initializer import rbm_linear_sequential_init
from src.nodes.metrics import update_metrics

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        # transforms.Normalize((0.5,), (0.5,)),
    ]
)


def get_classifier_model(features: List[int] = cc.NONE, loaded_model: OrderedDict = cc.NONE):
    model = Classifier(features=features)
    if not loaded_model == cc.NONE:
        model.load_state_dict(loaded_model)
    return model


def rbm_init_classifier(
    model,
    train_loader,
    device,
    is_model_initialized,
    preprocessing,
    rbm_epoch=1,
    rbm_type=rt.RBM,
    rbm_init_type=rit.IN_LAYER_ORDER,
):
    if not is_model_initialized:
        model.seq = rbm_linear_sequential_init(
            model.seq, train_loader, device, preprocessing, rbm_epoch, rbm_type, rbm_init_type
        )
    return model


def get_mnist_dataset(
    torch_dataset_path: str, train: bool = True, download: bool = False, transform=transform
) -> torch.utils.data.Dataset:
    return datasets.MNIST(root=torch_dataset_path, train=train, download=download, transform=transform)


def get_small_mnist_datasets(
    torch_dataset_path: str, train: bool = True, download: bool = False, transform=transforms.ToTensor()
) -> (torch.utils.data.Dataset, torch.utils.data.Dataset):
    dataset = datasets.MNIST(root=torch_dataset_path, train=train, download=download, transform=transform)
    return torch.utils.data.random_split(dataset, [1500, 150, 58350])[:-1]


def train_mnist_classifier(
    model, optimizer, loss_fn, train_loader, device, preprocessing=None, metrics: Dict[str, Any] = None
):
    time_start = time()
    model = model.train().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    average_loss = Average()
    for input, labels in train_loader:
        if preprocessing is not None:
            input = preprocessing(input)
        input = input.to(device).to(torch.double)
        optimizer.zero_grad()
        output = model(input)
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()
        average_loss.add(loss.item())
    time_end = time() - time_start
    data = {
        mov.TRAIN_AVERAGE_LOSS: {
            mc.VALUE: average_loss.avg,
            mc.BEST: mc.DOWN,
        },
    }
    return time_end, update_metrics(metrics, data), optimizer, model


def test_mnist_classifier(model, loss_fn, test_loader, device, preprocessing=None, metrics: Dict[str, Any] = None):
    time_start = time()
    model = model.train().to(device)
    with torch.no_grad():
        average_loss = Average()
        y_pred, y_true = [], []
        for input, labels in test_loader:
            if preprocessing is not None:
                input = preprocessing(input)
            input = input.to(device).to(torch.double)
            output_pred = model(input)
            y_true.append(labels)
            res = []
            for batch in output_pred:
                res.append(torch.argmax(batch).item())
            y_pred.append(torch.tensor(res))
            loss = loss_fn(output_pred, labels)
            average_loss.add(loss.item())
    time_end = time() - time_start
    data = {
        mov.TEST_AVERAGE_LOSS: {
            mc.VALUE: average_loss.avg,
            mc.BEST: mc.DOWN,
        },
    }
    return time_end, update_metrics(metrics, data), y_true, y_pred
