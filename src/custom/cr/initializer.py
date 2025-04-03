from torch import nn
from copy import deepcopy

import torch
from core.models import Classifier


class CumulativeRuleModel(nn.Module):
    def __init__(self, model):
        super(CumulativeRuleModel, self).__init__()
        self.forward_layers = nn.Sequential(*[item for item in model.model])
        self.reverse_layers = []
        for layer in reversed([item for item in model.model][:-1]):
            if isinstance(layer, nn.Linear):
                tmp = nn.Linear(in_features=layer.out_features, out_features=layer.in_features, dtype=torch.float64)
                layer = tmp
            self.reverse_layers.append(deepcopy(layer))
        self.reverse_layers.append(nn.ReLU())
        self.reverse_layers = nn.Sequential(*self.reverse_layers)

    def forward_step(self, data, model_layers, preprocess):
        results = []
        layers = []
        for layer in model_layers:
            layers.append(layer)
            if layer.__class__ in [nn.ReLU, nn.LeakyReLU, nn.Softmax]:
                tmp_model = nn.Sequential(*layers)
                data1 = preprocess(data)
                results.append(tmp_model(data1))
        return results

    def forward(self, x_0):
        forward_0 = self.forward_step(x_0, self.forward_layers, lambda x: x)
        middle_0 = forward_0[-1]
        reverse_0 = self.forward_step(x_0, self.reverse_layers, nn.Sequential(*self.forward_layers))
        x_1 = reverse_0[-1]
        forward_1 = self.forward_step(x_1, self.forward_layers, lambda x: x)
        reverse_1 = self.forward_step(x_1, self.reverse_layers, nn.Sequential(*self.forward_layers))
        return forward_0, reverse_0, middle_0, forward_1, reverse_1


class CumulativeRuleInitializer:
    def __init__(self, trainer, device, lr, epochs=None):
        self.trainer = trainer
        self.epochs = epochs
        self.device = device
        self.lr = lr

    def update_params(self, model: CumulativeRuleModel, y_forward_0, y_reverse_0, y_middle, y_forward_1, y_reverse_1, target, loss):
        i = 0
        mse = torch.nn.MSELoss()
        model_loss = loss(y_middle, target)
        for layer in model.forward_layers:
            if isinstance(layer, nn.Linear):
                if layer.weight.grad is not None:
                    layer.weight.grad.detach_()
                    layer.weight.grad.zero_()
                if layer.bias.grad is not None:
                    layer.bias.grad.detach_()
                    layer.bias.grad.zero_()
                error = mse(y_forward_1[i], y_forward_0[i]) + model_loss
                error.backward(retain_graph=True)
                layer.weight.data -= self.trainer.lr * layer.weight.grad
                layer.bias.data -= self.trainer.lr * layer.bias.grad
                i += 1

        i = 0
        for layer in model.reverse_layers:
            if isinstance(layer, nn.Linear):
                if layer.weight.grad is not None:
                    layer.weight.grad.detach_()
                    layer.weight.grad.zero_()
                if layer.bias.grad is not None:
                    layer.bias.grad.detach_()
                    layer.bias.grad.zero_()
                error = mse(y_reverse_1[i], y_reverse_0[i])
                error.backward(retain_graph=True)
                layer.weight.data -= self.trainer.lr * layer.weight.grad
                layer.bias.data -= self.trainer.lr * layer.bias.grad
                i += 1
        return model, model_loss


    def restore_model(self, model: CumulativeRuleModel):
        restored_model = Classifier([1, 1])
        restored_model.model = model.forward_layers
        return restored_model

    def __call__(self, model):
        model = CumulativeRuleModel(model)
        for epoch in range(self.epochs):
            avg_loss = 0
            for batch in self.trainer.train_loader:
                input_, target = self.trainer.get_data(batch)
                y_forward_0, y_reverse_0, y_middle, y_forward_1, y_reverse_1 = model.forward(input_)
                model, loss = self.update_params(model, y_forward_0, y_reverse_0, y_middle, y_forward_1, y_reverse_1, target, self.trainer.loss)
                avg_loss += loss.item()
        return self.restore_model(model)
