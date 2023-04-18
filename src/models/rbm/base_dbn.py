from old.model.layers.reshape import Reshape

# from src.model.pretrain.base_rbm import *
from torch import nn

from src.model.layers.flatten import Flatten


class BaseDBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.dbn_layer_types = (
            nn.Linear,
            nn.Conv1d,
            nn.Conv2d,
            nn.ConvTranspose1d,
            nn.ConvTranspose2d,
        )
        self.dbn_layer_types_no_param = (
            Reshape,
            Flatten,
            nn.MaxPool1d,
            nn.MaxPool2d,
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.Dropout,
            nn.Dropout2d,
        )
        self.dbn_layer_activation = (
            nn.Sigmoid,
            nn.Tanh,
            nn.ReLU,
            nn.LeakyReLU,
            nn.Softmax,
        )

    def forward(self, in_features, *args, **kwargs):
        pass

    def _create_rbm(self, layer):
        pass

    def next_layer(self):
        pass

    def _saved_trained_layer(self):
        pass

    def optimizer_step(self):
        pass

    def optimizer_zero_grad(self):
        pass

    def get_trained_layers(self):
        pass

    def parse_sequential(self, sequential_model):

        list_modules = [module for module in sequential_model.modules()][1:]

        layers = []
        i = 0
        while i < len(list_modules):
            layer_act = {}
            if type(list_modules[i]) in self.dbn_layer_types:
                layer_act["layer"] = list_modules[i]
                layer_act["f"] = None
                layer_act["layer_fn"] = None
                if i + 1 != len(list_modules) and type(list_modules[i + 1]) in self.dbn_layer_activation:
                    layer_act["f"] = list_modules[i + 1]
                    i += 1
                else:
                    layer_act["f"] = None
            elif type(list_modules[i]) in self.dbn_layer_types_no_param:
                layer_act["layer_fn"] = list_modules[i]
            else:
                print(
                    'Error. Check you model architecture. Layer type "{}" not recognized'.format(type(list_modules[i]))
                )
                exit(1)
            layers.append(layer_act)
            i += 1
        return layers
