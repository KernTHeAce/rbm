from collections import OrderedDict
from time import time
from typing import Any, Dict, List

import torch

from src.common.const import CommonConst as cc
from src.common.const import MetricConst as mc
from src.common.const import MetricsOutputValues as mov
from src.common.const import RBMInitTypes as rit
from src.common.const import RBMTypes as rt
from src.common.utils.average import Average
from src.models.autoencoder.ae import AE
from src.models.rbm.manual_linear_rbm_initializer import rbm_linear_sequential_init
from src.nodes.metrics import update_metrics


def get_ae_model(features: List[int] = cc.NONE, loaded_model: OrderedDict = cc.NONE):
    model = AE(features=features)
    if not loaded_model == cc.NONE:
        model.load_state_dict(loaded_model)
    return model


def rbm_init_ae(
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
        model.encoder = rbm_linear_sequential_init(
            model.encoder, train_loader, device, preprocessing, rbm_epoch, rbm_type, rbm_init_type
        )
        model.decoder = rbm_linear_sequential_init(
            model.decoder,
            train_loader,
            device,
            preprocessing,
            rbm_epoch,
            rbm_type,
            rbm_init_type,
            base_modules=model.encoder,
        )
    return model


def train_ae(model, optimizer, loss_fn, train_loader, device, preprocessing=None, metrics: Dict[str, Any] = None):
    time_start = time()
    model = model.train().to(device)
    # import os
    # if os.path.exists("data.txt"):
    #     with open("data1.txt", "w+") as f:
    #         f.write(str(model.encoder[0].bias.data))
    #     raise Exception("lol")
    # with open("data.txt", "w+") as f:
    #     f.write(str(model.encoder[0].bias.data))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)



    average_loss = Average()
    for data in train_loader:
        if len(data) != 2:
            input = data
        else:
            input, _ = data
        if preprocessing is not None:
            input = preprocessing(input)
        input = input.to(device).to(torch.double)

        input_encoded, input_decoded = model(input)
        loss = loss_fn(input_decoded, input)

        loss.backward()
        optimizer.step()

        average_loss.add(loss.item())
        optimizer.zero_grad()
    time_end = time() - time_start
    data = {
        mov.TRAIN_AVERAGE_LOSS: {
            mc.VALUE: average_loss.avg,
            mc.BEST: mc.DOWN,
        },
    }
    return time_end, update_metrics(metrics, data), optimizer, model


def test_ae(model, loss_fn, test_loader, device, preprocessing=None, metrics: Dict[str, Any] = None):
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
            input = input.to(device).to(torch.double)
            input_encoded, input_decoded = model(input)
            y_true.append(input)
            y_pred.append(input_decoded)

            loss = loss_fn(input_decoded, input)
            average_loss.add(loss.item())
    time_end = time() - time_start
    data = {
        mov.TEST_AVERAGE_LOSS: {
            mc.VALUE: average_loss.avg,
            mc.BEST: mc.DOWN,
        },
    }
    return time_end, update_metrics(metrics, data), y_true, y_pred
