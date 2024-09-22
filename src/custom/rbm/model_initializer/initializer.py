from typing import Any, Dict, List

from torch.nn import ReLU, Sequential

from .const import ParserConst as pc
from .rbm.rbm_adaptive_lr import LayerRbmAdaptiveLrInitializer
from .sequential_parser import SequentialParser


class ModelRBMInitializer:
    def __init__(
        self, trainer, device, lr, epochs=None, grad_min_max=(-100, 100), grad_clipping=False, adaptive_lr=False
    ):
        self.adaptive_lr = adaptive_lr
        self.trainer = trainer
        self.epochs = epochs
        self.device = device
        self.lr = lr
        self.grad_min_max = grad_min_max
        self.use_grad_clipping = grad_clipping

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
            for batch in self.trainer.train_loader:
                for i in range(len(layers)):
                    if not isinstance(layers[i][pc.FUNC], ReLU):
                        continue
                    rbm = LayerRbmAdaptiveLrInitializer(
                        layers[i][pc.LAYER],
                        layers[i][pc.FUNC],
                        self.lr,
                        self.grad_min_max,
                        biases[i],
                        self.device,
                        self.adaptive_lr,
                        self.trainer.train_loader.batch_size,
                        use_grad_clipping=self.use_grad_clipping,
                    )
                    input_, _ = self.trainer.get_data(batch)
                    if i != 0:
                        pretrained_model = Sequential(*self.layer_list_preprocess(layers[:i]))
                        input_ = pretrained_model(input_)
                        123
                    output = rbm.forward(input_)

                    if output is None:
                        return None
                    layers[i][pc.LAYER], biases[i] = rbm.get_trained_layer(get_bias=True)
        return Sequential(*self.layer_list_preprocess(layers))
