from torch import nn
from torch.nn import Module

from src.models.base_ae import BaseAE


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
            # encoder_modules.append(nn.Linear(in_features=in_features, out_features=out_features))
            encoder_modules.append(nn.Linear(in_features=in_features, out_features=out_features).double())
            # encoder_modules.append(nn.LeakyReLU())
            encoder_modules.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_modules)

        decoder_modules = []
        for i, (in_features, out_features) in enumerate(zip(self.features[::-1][:-1], self.features[::-1][1:])):
            # decoder_modules.append(nn.Linear(in_features=in_features, out_features=out_features))
            decoder_modules.append(nn.Linear(in_features=in_features, out_features=out_features).double())
            # decoder_modules.append(nn.LeakyReLU())
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

    # TODO move to BaseAE
    def from_dbn(self, trained_layers):
        encoder, decoder = [module for module in self.modules() if type(module) == nn.Sequential]
        encoder = [module for module in encoder.modules()][1:]
        decoder = [module for module in decoder.modules()][1:]
        assert len(trained_layers) == len(encoder + decoder)

        # First read encoded trained layers
        i = 0
        encoder_layers = nn.ModuleList()
        while i < len(encoder):
            encoder_layers.append(trained_layers[i])
            i += 1

        # Then read decoded trained layers
        decoder_layers = nn.ModuleList()
        while i < len(trained_layers):
            decoder_layers.append(trained_layers[i])
            i += 1

        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)
