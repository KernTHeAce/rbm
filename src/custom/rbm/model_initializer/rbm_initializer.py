from typing import Any, Dict, List

from torch.nn import ReLU, Sequential
from torch.optim import Adam

from .const import ParserConst as pc
from .rbm.rbm_adaptive_lr import LayerRbmAdaptiveLrInitializer
from .sequential_parser import SequentialParser


parser = SequentialParser()


class ModelRBMInitializer:
    def __init__(
        self, trainer, device, lr, epochs=1, grad_min_max=(-100, 100), grad_clipping=False, adaptive_lr=False, semisupervised_learning=False
    ):
        self.adaptive_lr = adaptive_lr
        self.trainer = trainer
        self.epochs = epochs
        self.device = device
        self.lr = lr
        self.grad_min_max = grad_min_max
        self.use_grad_clipping = grad_clipping
        self.semisupervised_learning = semisupervised_learning

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

    def adam_batch(self, layers, input_, target):
        model = Sequential(*self.layer_list_preprocess(layers))
        optimizer = Adam(model.parameters(), lr=1e-3)
        optimizer.zero_grad()
        output = model(input_)
        loss = self.trainer.loss(output, target)
        loss.backward()
        optimizer.step()
        return parser.get_layers(model)

    def adam_epoch(self, layers):
        model = Sequential(*self.layer_list_preprocess(layers))
        self.trainer.init_optimizer(model)
        model = self.trainer.epoch(model)[0]
        return parser.get_layers(model)

    def __call__(self, model):
        layers = parser.get_layers(model.model)
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
                        self.device,
                        self.adaptive_lr,
                        self.trainer.train_loader.batch_size,
                        use_grad_clipping=self.use_grad_clipping,
                    )
                    input_,  = self.trainer.get_data(batch)
                    if i != 0:
                        pretrained_model = Sequential(*self.layer_list_preprocess(layers[:i]))
                        input_ = pretrained_model(input_)
                    output = rbm.forward(input_)

                    if output is None:
                        return None
                    layers[i][pc.LAYER] = rbm.get_trained_layer()

                if self.semisupervised_learning:
                    input_, target = self.trainer.get_data(batch)
                    layers = self.adam_batch(layers, input_, target)
        return Sequential(*self.layer_list_preprocess(layers))
