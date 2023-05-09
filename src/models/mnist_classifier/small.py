from torch import nn


class Classifier(nn.Module):
    def __init__(self, features):
        super(Classifier, self).__init__()
        self.features = features
        modules = []
        for i, (in_features, out_features) in enumerate(zip(self.features[:-1], self.features[1:])):
            modules.append(nn.Linear(in_features=in_features, out_features=out_features).double())
            modules.append(nn.ReLU())
        del modules[-1]
        self.seq = nn.Sequential(*modules)

    def forward(self, x):
        # x = img.view(-1, 28*28)
        result = self.seq(x)
        return result
