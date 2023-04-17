from torch.utils.data import DataLoader
import pandas as pd
from src.models.autoencoder.ae import AE
from typing import List
import torch
from src.common.utils.average import Average
from time import time

from src.models.rbm.manual_linear_rbm_initializer import rbm_linear_sequential_init
from collections import OrderedDict

from src.common.const import CommonConst as cc

from .datasets import CSVSoccerDataset


def csv_to_data_loader(csv_file: pd.DataFrame, batch_size: int = 1, shuffle: bool = False):
    dataset = CSVSoccerDataset(csv_file)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def get_device(is_cuda: bool = False):
    return torch.device("cpu") if not is_cuda else torch.device("cuda:0")


def get_ae_model(features: List[int] = cc.NONE, loaded_model: OrderedDict = cc.NONE):
    model = AE(features=features)
    if not loaded_model == cc.NONE:
        model.load_state_dict(loaded_model)
    return model


def get_loss():
    return torch.nn.MSELoss()


def get_optimizer(model, lr, loaded_optim: OrderedDict = cc.NONE):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if not loaded_optim == cc.NONE:
        optimizer.load_state_dict(loaded_optim)
    return optimizer


def rbm_init_ae(model, train_loader, device):
    model.encoder = rbm_linear_sequential_init(model.encoder, train_loader, device)
    model.decoder = rbm_linear_sequential_init(model.decoder, train_loader, device, base_modules=model.encoder)
    return model


def one_epoch_ae_train(model,
                       optimizer,
                       loss_fn,
                       train_loader,
                       device,
                       preprocessing=None):
    time_start = time()
    model = model.train().to(device)
    average_loss = Average()
    for data in train_loader:
        if len(data) != 2:
            input = data
        else:
            input, _ = data
        if preprocessing is not None:
            input = preprocessing(input)
        input = input.to(device)
        optimizer.zero_grad()
        input_encoded, input_decoded = model(input)
        loss = loss_fn(input_decoded, input)

        loss.backward()
        optimizer.step()
        average_loss.add(loss.item())
    time_end = time() - time_start
    return time_end, average_loss.avg, optimizer


def test(model, loss_fn, test_loader, device, preprocessing=None):
    time_start = time()
    model = model.train().to(device)
    with torch.no_grad():
        average_loss = Average()
        y_pred, y_true = [], []
        for data in test_loader:
            if len(data) != 2:
                input = data
            else:
                input, _ = data

            if preprocessing is not None:
                input = preprocessing(input)
            input = input.to(device)
            input_encoded, input_decoded = model(input)
            y_true.append(input)
            y_pred.append(input_decoded)

            loss = loss_fn(input_decoded, input)
            average_loss.add(loss.item())
    time_end = time() - time_start
    return time_end, average_loss.avg, y_true, y_pred
