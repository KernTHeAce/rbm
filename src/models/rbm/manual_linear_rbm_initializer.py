from src.common.utils.sequential_parser import SequentialParser
from src.common.const import ParserConst as pc
from .rbm_manual_linear import RBMManualLinearCR
from torch.nn import Sequential


def rbm_linear_sequential_init(sequential, train_loader, device, epochs: int = 1, base_modules=None):
    parser = SequentialParser()
    layers = parser.get_layers(sequential)
    result_modules = []
    for layer_index, layer in enumerate(layers):
        rbm = RBMManualLinearCR(layer=layer[pc.LAYER], f=layer[pc.FUNC], device=device)
        for epoch in range(epochs):
            for data in train_loader:
                data = data.to(device)
                data = data if base_modules is None else base_modules(data)
                if layer_index != 0:
                    pretrained_model = Sequential(*result_modules)
                    data = pretrained_model(data)
                rbm.forward(data)
        result_modules.append(rbm.get_trained_layer())
        result_modules.append(layer[pc.FUNC])
    return Sequential(*result_modules)
