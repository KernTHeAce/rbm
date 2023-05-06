from collections import OrderedDict
from time import time
from typing import Any, Dict, List

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms

from src.common.const import CommonConst as cc
from src.common.const import MetricConst as mc
from src.common.const import MetricsOutputValues as mov
from src.common.utils.average import Average
from src.models.autoencoder.ae import AE
from src.models.mnist_classifier.small import Classifier
from src.models.rbm.manual_linear_rbm_initializer import rbm_linear_sequential_init
from src.nodes.metrics import update_metrics

from .datasets import CSVSoccerDataset

def prepr(input):
    return input.view(input.size()[0], -1)


def one_epoch_mnist_classifier_train(
    model, optimizer, loss_fn, train_loader, device, preprocessing=None, metrics: Dict[str, Any] = None
):
    time_start = time()
    model = model.train().to(device)
    average_loss = Average()
    for input, labels in train_loader:
        if preprocessing is not None:
            input = prepr(input)
        input = input.to(device).to(torch.double)
        optimizer.zero_grad()
        output_pred = model(input)
        loss = loss_fn(output_pred, labels)

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


def test(model, loss_fn, test_loader, device, preprocessing=None, metrics: Dict[str, Any] = None):
    time_start = time()
    model = model.train().to(device)
    with torch.no_grad():
        average_loss = Average()
        y_pred, y_true = [], []
        for input, labels in test_loader:
            if preprocessing is not None:
                input = prepr(input)
            input = input.to(device).to(torch.double)
            output_pred = model(input)
            y_true.append(labels)
            res = []
            for batch in output_pred:
                res.append(torch.argmin(batch).item())
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
