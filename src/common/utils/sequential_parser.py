from torch import nn

from src.common.const import ParserConst as pc


class SequentialParser:
    def __init__(self):
        super().__init__()
        self.layer_types = (
            nn.Linear,
            nn.Conv1d,
            nn.Conv2d,
            nn.ConvTranspose1d,
            nn.ConvTranspose2d,
        )
        self.layer_types_no_param = (
            nn.MaxPool1d,
            nn.MaxPool2d,
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.Dropout,
            nn.Dropout2d,
        )
        self.layer_activation = (nn.Sigmoid, nn.Tanh, nn.ReLU, nn.LeakyReLU, nn.Softmax)

    def is_activation(self, module):
        return type(module) in self.layer_activation

    def get_layers(self, sequential_model):
        list_modules = [module for module in sequential_model.modules()][1:]
        layers = []
        i = 0
        while i < len(list_modules):
            layer_act = {}
            if type(list_modules[i]) in self.layer_types:
                layer_act[pc.LAYER] = list_modules[i]
                layer_act[pc.FUNC] = None
                layer_act[pc.LAYER_FN] = None
                if i + 1 != len(list_modules) and type(list_modules[i + 1]) in self.layer_activation:
                    layer_act[pc.FUNC] = list_modules[i + 1]
                    i += 1
                else:
                    layer_act[pc.FUNC] = None
            elif type(list_modules[i]) in self.layer_types_no_param:
                layer_act[pc.LAYER_FN] = list_modules[i]
            else:
                print(
                    'Error. Check you model architecture. Layer type "{}" not recognized'.format(type(list_modules[i]))
                )
                exit(1)
            layers.append(layer_act)
            i += 1
        return layers
