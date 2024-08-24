from .rbm import LayerRBMInitializer
from .adaptive_lr2 import AdaptiveLRCalculator

from math import sqrt

import torch
from torch import nn
from torch import Tensor


class LayerRbmAdaptiveLrInitializer(LayerRBMInitializer):
    def __init__(
        self,
        layer,
        activation,
        lr,
        t_out: Tensor = None,
        w_out: Tensor = None,
        device=torch.device("cuda:0"),
        is_lr_adaptive=True,
        **adaptive_lr_kwargs
    ):
        super(LayerRBMInitializer, self).__init__(layer, activation, lr, t_out, w_out, device)
        self.is_lr_adaptive = is_lr_adaptive
        self.lr_calculator = AdaptiveLRCalculator(**adaptive_lr_kwargs) if is_lr_adaptive else None

    def forward(self, x0: Tensor, is_training: bool = True):
        with torch.no_grad():
            s_yo = self.get_weight_sum(x0, self.w_in, self.t_in)
            y0 = self.f(s_yo)
            s_x1 = self.get_weight_sum(y0, self.w_out, self.t_out)
            x1 = self.f(s_x1)
            s_y1 = self.get_weight_sum(x1, self.w_in, self.t_in)
            y1 = self.f(s_y1)
            if is_training:
                if self.is_lr_adaptive:
                    lr = self.lr_calculator()
                self.update_weights_biases(
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    s_x1=s_x1,
                    s_y1=s_y1,
                    lr=self.lr
                )
            return x0, y0, x1, y1
