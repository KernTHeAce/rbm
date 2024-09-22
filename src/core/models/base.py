from torch import nn


class BaseModel(nn.Module):
    def __init__(self, features: list, activation: nn.Module = nn.LeakyReLU):
        assert activation in [nn.ReLU, nn.LeakyReLU]
        super(BaseModel, self).__init__()
        self.features = features

        modules = []
        for i, (in_features, out_features) in enumerate(zip(self.features[:-1], self.features[1:])):
            modules.append(nn.Linear(in_features=in_features, out_features=out_features).double())
            modules.append(activation())
        modules = self.postprocess_modules(modules)
        self.model = nn.Sequential(*modules)

    @staticmethod
    def postprocess_modules(modules):
        return modules

    def forward(self, x):
        return self.model(x)

    def __str__(self):
        return f"Model model with features: {self.features}"
