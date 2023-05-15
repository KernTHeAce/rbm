import torch
from torch.nn import Sequential
from typing import List, Dict, Any

from common.const import CommonConst as cc
from common.const import ParserConst as pc
from common.const import RBMInitTypes as rit
from common.utils.sequential_parser import SequentialParser

from .rbm_manual_linear import RBMManualLinearCR


def rbm_linear_sequential_init(sequential, train_loader, device, preprocessing, epochs: int = 1, base_modules=None, type: str = rit.IN_LAYER_ORDER):
    if type == rit.IN_DATA_ORDER:
        return rbm_linear_sequential_init_in_data_order(sequential, train_loader, device, preprocessing, epochs, base_modules)
    elif type == rit.IN_LAYER_ORDER:
        return rbm_linear_sequential_init_in_layer_order(sequential, train_loader, device, preprocessing, epochs, base_modules)


def rbm_linear_sequential_init_in_data_order(sequential, train_loader, device, preprocessing, epochs: int = 1, base_modules=None):
    parser = SequentialParser()
    layers = parser.get_layers(sequential)
    result_modules = []
    for layer_index, layer in enumerate(layers):
        rbm = RBMManualLinearCR(layer=layer[pc.LAYER], f=layer[pc.FUNC], device=device)
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
                if layer_index != 0:
                    pretrained_model = Sequential(*result_modules)
                    input = pretrained_model(input)
                rbm.forward(input)
        result_modules.append(rbm.get_trained_layer())
        if layer[pc.FUNC]:
            result_modules.append(layer[pc.FUNC])
    return Sequential(*result_modules)


def rbm_linear_sequential_init_in_layer_order(sequential, train_loader, device, preprocessing, epochs: int = 1, base_modules=None):
    parser = SequentialParser()
    layers = parser.get_layers(sequential)
    for epoch in range(epochs):
        for data in train_loader:
            for i in range(len(layers)):
                rbm = RBMManualLinearCR(layer=layers[i][pc.LAYER], f=layers[i][pc.FUNC], device=device)
                # data preprocess
                if len(data) != 2:
                    input = data
                else:
                    input, _ = data
                if preprocessing and preprocessing != cc.NONE:
                    input = preprocessing(input)
                input = input.to(device).to(torch.double)
                input = input if base_modules is None else base_modules(input)

                layers[i][pc.LAYER] = rbm.get_trained_layer()
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
