from .sequential_parser import SequentialParser

from typing import Any, Dict, List

import torch
from torch.nn import Sequential

from src.common.const import CommonConst as cc
from src.common.const import ParserConst as pc
from src.common.const import RBMInitTypes as rit
from src.common.const import RBMTypes as rt

from .rbm import LayerRBMInitializer
from src import DEVICE


class ModelRBMInitializer:
    def __init__(self, train_loader, epochs, device, adaptive_lr=False):
        self.adaptive_lr = adaptive_lr
        self.loader = train_loader
        self.epochs = epochs
        self.device = device

    @staticmethod
    def layer_list_preprocess(layers: List[Dict[str, Any]]):
        res = []
        for item in layers:
            res.append(item[pc.LAYER])
            if item[pc.FUNC]:
                res.append(item[pc.FUNC])
        return res

    def __call__(self, model):
        parser = SequentialParser()
        layers = parser.get_layers(model.model)
        biases = [None] * len(layers)
        weights = [None] * len(layers)
        for epoch in range(self.epochs):
            for data in self.loader:
                for i in range(len(layers)):
                    rbm = LayerRBMInitializer(
                        layer=layers[i][pc.LAYER],
                        f=layers[i][pc.FUNC],
                        t_out=biases[i],
                        w_out=weights[i],
                    )
                    input_ = data.to(self.device)
                    if i != 0:
                        pretrained_model = Sequential(*self.layer_list_preprocess(layers[:i]))
                        input_ = pretrained_model(input_)
                    rbm.forward(input_)
                    layers[i][pc.LAYER] = rbm.get_trained_layer()
        return Sequential(*self.layer_list_preprocess(layers))
