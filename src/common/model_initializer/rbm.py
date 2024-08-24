from math import sqrt
from typing import Tuple

import torch
from torch import nn

from src.models.rbm.base_rbm import BaseRBM


class LayerRBMInitializer(BaseRBM):

    def __init__(
        self,
        layer,
        f=nn.LeakyReLU(),
        lr_min_max: Tuple[float, float] = None,
        t_out: torch.Tensor = None,
        w_out: torch.Tensor = None,
        commulative_rule: bool = False,
        device=torch.device("cuda:0"),
    ):
        super(BaseRBM, self).__init__()

        assert isinstance(layer, nn.Linear)

        self.device = device
        self.w_in = layer.weight.data.clone().to(device)
        self.t_in = layer.bias.data.clone().to(device)
        self.w_in.requires_grad_(True)
        self.out_features, self.in_features = self.w_in.size()
        self.w_out, self.t_out = self.init_param(w_out, t_out, commulative_rule)
        self.f, self.f_ = self.activation(f, using_derivative=True)
        self.lr_min_max = lr_min_max if lr_min_max else (1e-8, 1e-2)

    def init_param(self, w_out, t_out, commulative_rule):
        stdv = 1.0 / sqrt(self.in_features)
        w = self.w_in.t().data.clone()
        b = (
            torch.Tensor(self.in_features).uniform_(-stdv, stdv).to(self.device).to(torch.double)
            if t_out is None
            else t_out
        )
        if commulative_rule:
            w = (
                torch.Tensor(self.in_features, self.out_features).uniform_(-stdv, stdv).to(self.device).to(torch.double)
                if w_out is None
                else w_out
            )
        return w, b

    def adaptive_lr_no_bias(self, x0, y0, s_x1, x1, s_y1, y1):
        """For LeakyReLU function only"""

        if not isinstance(self.f, nn.LeakyReLU):
            return None, None

        with torch.no_grad():
            rj = self.f_(s_y1)
            ri = self.f_(s_x1)
            rj_next = self.f_(y0)
            ri_next = self.f_(x0)
            bj = (y1 - y0) * rj * (x1.square()).sum()
            bi = (x1 - x0) * ri * (y0.square()).sum()
            lr_y_upper = (s_y1 * rj_next - y0) * bj * rj_next
            lr_y_down = torch.square(rj_next * bj)
            lr_y = lr_y_upper.sum() / lr_y_down.sum()
            lr_x_upper = (s_x1 * ri_next - x0) * bi * ri_next
            lr_x_down = torch.square(ri_next * bi)
            lr_x = lr_x_upper.sum() / lr_x_down.sum()
            return self.min_max_lr(lr_x), self.min_max_lr(lr_y)

    def min_max_lr(self, lr: float):
        if self.lr_min_max[1] > lr > self.lr_min_max[0]:
            return lr
        else:
            if lr > self.lr_min_max[1]:
                return self.lr_min_max[1]
            else:
                return self.lr_min_max[0]

    def min_max_lr_every(self, lr: torch.Tensor):
        lr[lr == torch.NoneType] = self.lr_min_max[1]
        lr[lr > self.lr_min_max[1]] = self.lr_min_max[1]
        lr[lr < self.lr_min_max[0]] = self.lr_min_max[0]
        return lr

    def forward(
        self,
        x0: torch.Tensor,
        lr: float = 1e-3,
        lr_auto: bool = False,
        lr_const: int = None,
        is_training: bool = True,
    ):
        with torch.no_grad():
            s_yo = self.get_weight_sum(x0, self.w_in, self.t_in)
            y0 = self.f(s_yo)
            s_x1 = self.get_weight_sum(y0, self.w_out, self.t_out)
            x1 = self.f(s_x1)
            s_y1 = self.get_weight_sum(x1, self.w_in, self.t_in)
            y1 = self.f(s_y1)
            if is_training:
                self.update_weights_biases(
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    s_x1=s_x1,
                    s_y1=s_y1,
                    lr=lr,
                    lr_auto=lr_auto,
                    lr_const=lr_const,
                )
            return x0, y0, x1, y1

    @staticmethod
    def get_weight_sum(in_features, w: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.add(torch.matmul(in_features, w.t()), t)

    def update_weights_biases(
        self,
        x0: torch.Tensor,
        y0: torch.Tensor,
        x1: torch.Tensor,
        y1: torch.Tensor,
        s_x1: torch.Tensor,
        s_y1: torch.Tensor,
        lr: float,
        lr_auto: bool = False,
        lr_const: int = None,
    ):
        w_in_grad, t_in_grad = self.get_weight_bias_grad(output=y1, reference=y0, output_ws=s_y1, middle_output=x1)
        w_out_grad, t_out_grad = self.get_weight_bias_grad(output=x1, reference=x0, output_ws=s_x1, middle_output=y0)
        lr_x, lr_y = self.get_lr(
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
            s_x1=s_x1,
            s_y1=s_y1,
            lr=lr,
            lr_auto=lr_auto,
            lr_const=lr_const,
        )
        self.w_in = self.w_in - lr_y * w_in_grad
        self.t_in = self.t_in - lr_y * t_in_grad
        self.w_out = self.w_out - lr_x * w_out_grad
        self.t_out = self.t_out - lr_x * t_out_grad

    def get_weight_bias_grad(
        self,
        output: torch.Tensor,
        reference: torch.Tensor,
        output_ws: torch.Tensor,
        middle_output: torch.Tensor,
    ):
        grad_part = (output - reference) * self.f_(output_ws)
        w_grad = torch.matmul(grad_part.t(), middle_output)
        b_grad = grad_part.sum(dim=0)
        return w_grad, b_grad

    def get_lr(
        self,
        x0: torch.Tensor,
        y0: torch.Tensor,
        x1: torch.Tensor,
        y1: torch.Tensor,
        s_x1: torch.Tensor,
        s_y1: torch.Tensor,
        lr: float,
        lr_auto: bool = False,
        lr_const: int = None,
    ):
        if lr_auto:
            if not lr_const:
                lr_x, lr_y = self.adaptive_lr_bias(x0=x0, y0=y0, s_x1=s_x1, x1=x1, s_y1=s_y1, y1=y1)
            else:
                lr_x, lr_y = (lr, lr)
        else:
            lr_x, lr_y = (lr, lr)
        return lr_x, lr_y

    def adaptive_lr_bias(self, x0, y0, s_x1, x1, s_y1, y1):
        """For LeakyReLU function only"""

        with torch.no_grad():
            rj = self.f_(s_y1)
            ri = self.f_(s_x1)
            rj_next = self.f_(y0)
            ri_next = self.f_(x0)
            bj = (y1 - y0) * rj * (1 + (x1.square()).sum())
            bi = (x1 - x0) * ri * (1 + (y0.square()).sum())
            lr_y = ((s_y1 * rj_next - y0) * bj * rj_next).sum() / (torch.square(rj_next * bj)).sum()
            lr_x = ((s_x1 * ri_next - x0) * bi * ri_next).sum() / (torch.square(ri_next * bi)).sum()
            if self.lr_min_max is not None:
                return self.min_max_lr(lr_x), self.min_max_lr(lr_y)
            else:
                return lr_x, lr_y

    def get_trained_layer(self, get_bias: bool = False, get_weights: bool = False):
        layer = nn.Linear(in_features=self.in_features, out_features=self.out_features)
        layer.weight.data = self.w_in.data.clone()
        layer.bias.data = self.t_in.data.clone()
        results = [layer]
        if get_bias:
            results.append(self.t_out.data.clone())
        if get_weights:
            results.append(self.w_out.data.clone())
        return tuple(results) if len(results) > 1 else layer


