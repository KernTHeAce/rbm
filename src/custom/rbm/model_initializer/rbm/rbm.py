import math

import torch
from torch import Tensor, nn
from torch.nn.parameter import Parameter

from .base_rbm import BaseRBM


class LayerRBMInitializer:
    def __init__(
        self,
        layer,
        activation,
        lr,
        t_out: Tensor = None,
        device=torch.device("cuda:0"),
    ):
        # super(BaseRBM, self).__init__()
        assert isinstance(layer, nn.Linear)
        self.device = device
        self.lr = lr
        self.w_in = layer.weight.data.clone().to(device)
        self.t_in = layer.bias.data.clone().to(device)
        # self.w_in.requires_grad_(True)
        self.out_features, self.in_features = self.w_in.size()
        self.w_out, self.t_out = self.init_out_params(t_out)
        self.f, self.f_ = self.activation(activation, using_derivative=True)

    def init_out_params(self, t_out):
        w = self.w_in.t().data.clone()
        if t_out is not None:
            return w, t_out

        t = Parameter(torch.empty(self.in_features, device=self.device, dtype=None))  # initializing like pytorch
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(w)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(t, -bound, bound)
        return w, t

    def forward(self, x0: Tensor, is_training: bool = True):
        with torch.no_grad():
            s_y0 = self.get_weight_sum(x0, self.w_in, self.t_in)
            y0 = self.f(s_y0)
            s_x1 = self.get_weight_sum(y0, self.w_out, self.t_out)
            x1 = self.f(s_x1)
            s_y1 = self.get_weight_sum(x1, self.w_in, self.t_in)
            y1 = self.f(s_y1)
            if is_training:
                self.update_weights_biases(x0=x0, y0=y0, x1=x1, y1=y1, s_x1=s_x1, s_y1=s_y1, lr=self.lr)
            return x0, y0, x1, y1, s_x1, s_y1

    @staticmethod
    def get_weight_sum(in_features, w: Tensor, t: Tensor) -> Tensor:
        return torch.add(torch.matmul(in_features, w.t()), t)

    def update_weights_biases(
        self, x0: Tensor, y0: Tensor, x1: Tensor, y1: Tensor, s_x1: Tensor, s_y1: Tensor, lr: float
    ):
        w_in_grad = torch.matmul((y1 - y0).t(), self.f(x1)) + torch.matmul((x1 - x0).t(), self.f(y0)).t()
        t_in_grad = self.f(y1 - y0).sum(dim=0)
        t_out_grad = self.f(x1 - x0).sum(dim=0)
        self.w_in -= lr * w_in_grad
        self.t_in -= lr * t_in_grad
        self.w_out = self.w_in.t()
        self.t_out -= lr * t_out_grad

    def get_weight_bias_grad(self, output: Tensor, reference: Tensor, output_ws: Tensor, middle_output: Tensor):
        grad_part = (output - reference) * self.f_(output_ws)
        w_grad = torch.matmul(grad_part.t(), middle_output)
        b_grad = grad_part.sum(dim=0)
        return w_grad, b_grad

    def get_trained_layer(self, get_bias: bool = False):
        layer = nn.Linear(in_features=self.in_features, out_features=self.out_features)
        layer.weight.data = self.w_in.data.clone()
        layer.bias.data = self.t_in.data.clone()
        if get_bias:
            return layer, self.t_out.data.clone()
        return layer

    @staticmethod
    def activation(f, using_derivative):
        # assert isinstance(f, nn.LeakyReLU)
        if f is not None:
            if using_derivative:
                f_ = BaseRBM.derivative(f)
            else:
                f_ = lambda y: 1.0
        else:
            f = lambda s: s
            f_ = lambda y: 1.0
        return f, f_

    @staticmethod
    def derivative(f):
        if type(f) == nn.Sigmoid:
            f_ = lambda y: y * (1.0 - y)
        elif type(f) == nn.Tanh:
            f_ = lambda y: 1.0 - (y**2)
        elif type(f) == nn.ReLU:

            def _relu(y):
                yc = y.clone()
                yc[yc >= 0] = 1.0
                yc[yc < 0] = 0.0
                return yc

            f_ = _relu
        elif type(f) == nn.LeakyReLU:

            def _leaky_relu(y):
                yc = y.clone()
                yc[yc >= 0] = 1.0
                yc[yc < 0] = f.negative_slope
                return yc

            f_ = _leaky_relu
        else:
            print('Error. Activation function "{}" is not support!'.format(type(f)))
            exit(1)
        return f_
