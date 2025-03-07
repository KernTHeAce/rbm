from typing import Any, Dict, List

from torch.nn import ReLU, Sequential
from torch.optim import Adam

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

    def _rbm_one_batch_forward(self, layers, input_):
        for i in range(len(layers)):
            if not isinstance(layers[i][pc.FUNC], ReLU):
                continue
            rbm = LayerRbmAdaptiveLrInitializer(
                layers[i][pc.LAYER],
                layers[i][pc.FUNC],
                self.lr,
                self.grad_min_max,
                self.device,
                self.adaptive_lr,
                self.trainer.train_loader.batch_size,
                use_grad_clipping=self.use_grad_clipping,
            )
            if i != 0:
                pretrained_model = Sequential(*self.layer_list_preprocess(layers[:i]))
                input_ = pretrained_model(input_)
            output = rbm.forward(input_)

            if output is None:
                return None
            layers[i][pc.LAYER] = rbm.get_trained_layer()
            return layers

    def __call__(self, model, loss):
        parser = SequentialParser()
        layers = parser.get_layers(model.model)
        for epoch in range(self.epochs):
            for batch in self.trainer.train_loader:
                input_, target = self.trainer.get_data(batch)
                layers = self._rbm_one_batch_forward(layers, input_)
                tmp_model = Sequential(*self.layer_list_preprocess(layers))
                optimizer = Adam(params=tmp_model.parameters(), lr=1e-3)
                output = tmp_model(input_)
                error = loss(output, target)
                error.backward()
                optimizer.step()
                layers = parser.get_layers(tmp_model)
        return Sequential(*self.layer_list_preprocess(layers))
