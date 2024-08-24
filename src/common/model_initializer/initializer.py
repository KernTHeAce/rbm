from typing import Any, Dict, List
from torch.nn import Sequential

from .sequential_parser_const import ParserConst as pc
from .rbm.rbm import LayerRBMInitializer
from .sequential_parser import SequentialParser


class ModelRBMInitializer:
    def __init__(self, train_loader, epochs, device, lr, adaptive_lr=False):
        self.adaptive_lr = adaptive_lr
        self.loader = train_loader
        self.epochs = epochs
        self.device = device
        self.lr = lr

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
        for epoch in range(self.epochs):
            for data in self.loader:
                for i in range(len(layers)):
                    rbm = LayerRBMInitializer(
                        layer=layers[i][pc.LAYER],
                        activation=layers[i][pc.FUNC],
                        t_out=biases[i],
                        lr=self.lr,
                        is_lr_adaptive=self.adaptive_lr,
                        batch_size=self.loader.batch_size,
                        batch_num=len(self.loader),
                    )
                    input_ = data.to(self.device)
                    if i != 0:
                        pretrained_model = Sequential(*self.layer_list_preprocess(layers[:i]))
                        input_ = pretrained_model(input_)
                    rbm.forward(input_)
                    layers[i][pc.LAYER], biases[i] = rbm.get_trained_layer(get_bias=True)
        return Sequential(*self.layer_list_preprocess(layers))
