from torch import nn
from torch.nn import Module


class Model(Module):
    def __init__(self, features: list):
        super(Model, self).__init__()
        self.features = features

        modules = []
        for i, (in_features, out_features) in enumerate(zip(self.features[:-1], self.features[1:])):
            modules.append(nn.Linear(in_features=in_features, out_features=out_features).double())
            modules.append(nn.LeakyReLU())
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)

    def __str__(self):
        return f"Model model with features: {self.features}"
