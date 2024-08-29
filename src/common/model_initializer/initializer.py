from typing import Any, Dict, List
from torch.nn import Sequential

from .const import ParserConst as pc
from .rbm.rbm_adaptive_lr import LayerRbmAdaptiveLrInitializer
from .sequential_parser import SequentialParser


class ModelRBMInitializer:
    def __init__(
            self, train_loader, epochs, device, lr, grad_min_max=(-100, 100), use_grad_clipping=False, adaptive_lr=False
    ):
        self.adaptive_lr = adaptive_lr
        self.loader = train_loader
        self.epochs = epochs
        self.device = device
        self.lr = lr
        self.grad_min_max = grad_min_max
        self.use_grad_clipping = use_grad_clipping

    def __bool__(self):
        return self.adaptive_lr is not None

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
            for i_, data in enumerate(self.loader):
                for i in range(len(layers)):
                    # print(epoch, i_, i)
                    rbm = LayerRbmAdaptiveLrInitializer(
                        layers[i][pc.LAYER],
                        layers[i][pc.FUNC],
                        self.lr,
                        self.grad_min_max,
                        biases[i],
                        self.device,
                        self.adaptive_lr,
                        self.loader.batch_size,
                        use_grad_clipping=self.use_grad_clipping
                    )
                    input_ = data.to(self.device)
                    if i != 0:
                        pretrained_model = Sequential(*self.layer_list_preprocess(layers[:i]))
                        input_ = pretrained_model(input_)
                    output = rbm.forward(input_)

                    if output is None:
                        return None
                    layers[i][pc.LAYER], biases[i] = rbm.get_trained_layer(get_bias=True)
        return Sequential(*self.layer_list_preprocess(layers))
