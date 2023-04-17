from abc import abstractmethod

from torch.nn import Module


class BaseAE(Module):
    @abstractmethod
    def set_encoder(cls, model: Module) -> None:
        pass

    @abstractmethod
    def set_decoder(cls, model: Module) -> None:
        pass

    @abstractmethod
    def get_encoder(cls):
        pass

    @abstractmethod
    def get_decoder(cls):
        pass

    def freeze(self):
        for param in self.parameters():
            param.required_grad = False

    # TODO add code for set new encoder & decoder values
    def from_dbn(self, trained_layers):
        encoder = self.get_encoder()
        decoder = self.get_decoder()
