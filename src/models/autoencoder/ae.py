from torch import nn
from torch.nn import Module

from src.models.autoencoder.base_ae import BaseAE


class AE(BaseAE):
    """
    Convolutional autoencoder for compress wyscout soccer data
    model structure example:
        Encoder:
            1. 196 -> 191
            2. 191 -> 95
        Decoder:
            1. 95 -> 191
            2. 191 -> 196
    """

    def __init__(self, features: list):
        super(AE, self).__init__()
        self.model_name = "ae"
        self.features = features

        encoder_modules = []
        for i, (in_features, out_features) in enumerate(zip(self.features[:-1], self.features[1:])):
            encoder_modules.append(nn.Linear(in_features=in_features, out_features=out_features).double())
            encoder_modules.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_modules)

        decoder_modules = []
        for i, (in_features, out_features) in enumerate(zip(self.features[::-1][:-1], self.features[::-1][1:])):
            decoder_modules.append(nn.Linear(in_features=in_features, out_features=out_features).double())
            decoder_modules.append(nn.ReLU())
        decoder_modules[-1] = nn.Sigmoid()
        self.decoder = nn.Sequential(*decoder_modules)

    def forward(self, x):
        x_encoded = self.encoder(x)
        x_decoded = self.decoder(x_encoded)
        return x_encoded, x_decoded

    def set_decoder(self, model: Module) -> None:
        self.decoder = model

    def set_encoder(self, model: Module) -> None:
        self.encoder = model

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def __str__(self):
        return f"AE model with features: {self.features}"
