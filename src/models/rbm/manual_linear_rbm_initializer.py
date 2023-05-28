from typing import Any, Dict, List

import torch
from torch.nn import Sequential

from src.common.const import CommonConst as cc
from src.common.const import ParserConst as pc
from src.common.const import RBMInitTypes as rit
from src.common.const import RBMTypes as rt
from src.common.utils.sequential_parser import SequentialParser

# from .rbm_manual_linear import RBMManualLinearCR
from .rbm_linear_commulative_rule import RBMLinearCR


def rbm_linear_sequential_init(
    sequential,
    train_loader,
    device,
    preprocessing,
    epochs,
    rbm_type: str = rt.RBM,
    rbm_init_type: str = rit.IN_LAYER_ORDER,
    base_modules=None,
):
    if rbm_type == rt.NO_RBM:
        return sequential

    if rbm_init_type == rit.IN_DATA_ORDER:
        return rbm_linear_sequential_init_in_data_order(
            sequential, train_loader, device, preprocessing, epochs, rbm_type, base_modules
        )
    elif rbm_init_type == rit.IN_LAYER_ORDER:
        return rbm_linear_sequential_init_in_layer_order(
            sequential, train_loader, device, preprocessing, epochs, rbm_type, base_modules
        )


def rbm_linear_sequential_init_in_data_order(
    sequential, train_loader, device, preprocessing, epochs, rbm_type, base_modules=None
):
    parser = SequentialParser()
    layers = parser.get_layers(sequential)
    result_modules = []
    biases = [None] * len(layers)
    for index, layer in enumerate(layers):
        rbm = RBMLinearCR(
            layer=layer[pc.LAYER],
            f=layer[pc.FUNC],
            device=device,
            t_out=biases[index],
            commulative_rule=rbm_type == rt.CRRBM,
        )
        for epoch in range(epochs):
            for data in train_loader:
                if len(data) != 2:
                    input = data
                else:
                    input, _ = data
                if preprocessing and preprocessing != cc.NONE:
                    input = preprocessing(input)
                input = input.to(device).to(torch.double)
                input = input if base_modules is None else base_modules(input)
                if index != 0:
                    pretrained_model = Sequential(*result_modules)
                    input = pretrained_model(input)
                rbm.forward(input)
        result_modules.append(rbm.get_trained_layer())
        if layer[pc.FUNC]:
            result_modules.append(layer[pc.FUNC])
    return Sequential(*result_modules)


def rbm_linear_sequential_init_in_layer_order(
    sequential, train_loader, device, preprocessing, epochs, rbm_type, base_modules=None
):
    parser = SequentialParser()
    layers = parser.get_layers(sequential)
    biases = [None] * len(layers)
    weights = [None] * len(layers)
    for epoch in range(epochs):
        for data in train_loader:
            for i in range(len(layers)):
                rbm = RBMLinearCR(
                    layer=layers[i][pc.LAYER],
                    f=layers[i][pc.FUNC],
                    device=device,
                    t_out=biases[i],
                    w_out=weights[i],
                    commulative_rule=rbm_type == rt.CRRBM,
                )
                # data preprocess
                if len(data) != 2:
                    input = data
                else:
                    input, _ = data
                if preprocessing and preprocessing != cc.NONE:
                    input = preprocessing(input)
                input = input.to(device).to(torch.double)
                input = input if base_modules is None else base_modules(input)
                if rbm_type == rt.RBM:
                    layers[i][pc.LAYER], biases[i] = rbm.get_trained_layer(get_bias=True)
                else:
                    layers[i][pc.LAYER], biases[i], weights[i] = rbm.get_trained_layer(get_bias=True, get_weights=True)
                if i != 0:
                    pretrained_model = Sequential(*layer_list_preprocess(layers[:i]))
                    input = pretrained_model(input)
                rbm.forward(input)
    return Sequential(*layer_list_preprocess(layers))


def layer_list_preprocess(layers: List[Dict[str, Any]]):
    res = []
    for item in layers:
        res.append(item[pc.LAYER])
        if item[pc.FUNC]:
            res.append(item[pc.FUNC])
    return res
