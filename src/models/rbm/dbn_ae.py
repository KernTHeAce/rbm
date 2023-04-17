from typing import Dict, List, Tuple

import torch
from torch import nn

from src.model.pretrain.base_dbn import BaseDBN


class DBNAE(BaseDBN):
    """
    Autoencoder model must have 2 Sequential modules: encoder and decoder
    """

    def __init__(
        self,
        autoencoder,
        rbm_cls,
        input_shape: Tuple[int, int],
        batch_size: int = 1,
        lr: float = 1e-3,
        use_min_max: bool = True,
        lr_min_max: List[float] = None,
        using_derivative: bool = True,
        using_optimizer: bool = True,
        rbm_biases: Dict[str, torch.Tensor] = None,
        device=torch.device("cuda:0"),
    ):
        super(DBNAE, self).__init__()

        self.input_shape = input_shape
        self.batch_size = batch_size
        self.lr = lr
        self.use_min_max = use_min_max
        self.lr_min_max = lr_min_max
        self.device = device
        self.using_derivative = using_derivative

        # Get model layers & names and indexes
        self.training_layers = self.parse_ae(autoencoder)

        # Current training layer
        self.training_index = -1

        # Get all strides and paddings for calculate output paddings
        # self.output_paddings = self._output_paddings(self.training_layers)

        # Create RBM for current training layer
        self.rbm = None
        self.rbm_cls = rbm_cls
        self.rbm_biases = rbm_biases if rbm_biases else {}

        # Create optimizer
        self.optimizer = None
        self.using_optimizer = using_optimizer

        # Trained layer
        self.trained_layers = nn.ModuleList()

    def parse_ae(self, model: nn.Module):
        encoder, decoder = [module for module in model.modules() if type(module) == nn.Sequential]
        encoder = self.parse_sequential(encoder)
        decoder = self.parse_sequential(decoder)

        return encoder + decoder

    def _create_rbm(self, layer: nn.Module):

        # Get size of input data for current RBM layer
        in_features = torch.randn(self.input_shape).to(self.device)
        if len(self.trained_layers) > 0:
            for trained_layer in self.trained_layers:
                with torch.no_grad():
                    in_features = trained_layer(in_features)

        # Create RBM for training layer
        rbm = self.rbm_cls(
            layer=layer["layer"],
            input_shape=in_features.shape,
            batch_size=self.batch_size,
            using_derivative=self.using_derivative,
            f=layer["f"],
            t_out=self.rbm_biases.get(self.training_index),
            use_min_max=self.use_min_max,
            lr_min_max=self.lr_min_max,
        )

        rbm.to(self.device)

        return rbm

    def forward(self, in_features, *args, **kwargs):
        if len(self.trained_layers) > 0:
            for layer in self.trained_layers:
                with torch.no_grad():
                    in_features = layer(in_features)
        return self.rbm(in_features, *args, **kwargs)

    def next_layer(self):
        self.training_index += 1

        # Check if we have not trained layers
        if self.training_index < self.layers_count():

            # Creating new RBM for next training layer
            layer = self.training_layers[self.training_index]

            # If we have RBM for previous layer saving their results as trained layer
            if self.rbm is not None:
                self._saved_trained_layer()
            if not layer["layer_fn"]:
                if not self.rbm:
                    # Just creating RBM
                    self.rbm = self._create_rbm(layer)
                    if self.using_optimizer:
                        self.optimizer = torch.optim.SGD(self.rbm.parameters(), lr=self.lr)
            else:
                self.trained_layers.append(layer["layer_fn"].to(self.device))
                return self.next_layer()
            return True
        else:
            if self.rbm:
                self._saved_trained_layer()
            self.training_index = -1
            return False

    def optimizer_zero_grad(self):
        if self.optimizer is not None:
            self.optimizer.zero_grad()

    def optimizer_step(self):
        if self.optimizer is not None:
            self.optimizer.step()

    def _saved_trained_layer(self):
        """Saved trained layer and reset RBM to None"""
        trained_layer = self.rbm.get_trained_layer()
        trained_layer = trained_layer.to(self.device)
        self.trained_layers.append(trained_layer)
        trained_layer_f = self.training_layers[self.training_index - 1]["f"]

        # If trained layer from previous RBM have activation function add them else pass
        if trained_layer_f is not None:
            self.trained_layers.append(trained_layer_f)
        self.rbm_biases[self.training_index - 1] = self.rbm.get_bias_out()
        self.rbm = None

    def layers_count(self):
        return len(self.training_layers)

    def get_trained_layers(self):
        return self.trained_layers

    def get_rbm_biases(self):
        return self.rbm_biases
