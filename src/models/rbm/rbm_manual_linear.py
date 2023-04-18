from math import sqrt
from typing import Tuple

import torch
from torch import nn

from src.models.rbm.base_rbm import BaseRBM

# The flag below controls whether to allow TF32 on matmul. This flag defaults to False
# in PyTorch 1.12 and later.
# torch.set_default_dtype(torch.double)
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
#
#
# # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
# torch.backends.cudnn.allow_tf32 = True


# TODO need to saving t_out to dbn for continue training by epoch_per_layer
class RBMManualLinearCR(BaseRBM):
    """Cumulative rule for Linear layer"""

    def __init__(
        self,
        layer,
        input_shape=None,
        batch_size=1,
        f=nn.Tanh(),
        using_derivative=True,
        use_min_max=True,
        lr_min_max: Tuple[float, float] = None,
        t_out: torch.Tensor = None,
        device=torch.device("cuda:0"),
    ):
        super(RBMManualLinearCR, self).__init__()

        assert isinstance(layer, nn.Linear)

        self.device = device
        self.w_in = layer.weight.data.clone().to(device)
        self.t_in = layer.bias.data.clone().to(device)
        self.w_in.requires_grad_(True)
        self.h_features, self.in_features = self.w_in.size()
        self.w_out, self.t_out = self.init_param(t_out)
        self.f, self.f_ = self.activation(f, using_derivative)
        self.using_derivative = using_derivative
        self.use_min_max = use_min_max
        self.lr_min_max = lr_min_max if lr_min_max else (1e-8, 1e-2)
        self.batch_size = batch_size

        # TODO
        self.epoch_printed = -1
        self.grads = {}

    def init_param(self, t_out: torch.Tensor = None):
        stdv = 1.0 / sqrt(self.in_features)
        # w = torch.Tensor(self.in_features, self.h_features).uniform_(-stdv, stdv).to(self.device)
        # b = torch.Tensor(self.in_features).uniform_(-stdv, stdv).to(self.device)

        w = self.w_in.t().data.clone()
        # w = self.w_in.rot90().data.clone()
        b = torch.Tensor(self.in_features).uniform_(-stdv, stdv).to(self.device) if t_out is None else t_out
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

    def _forward_no_bias(self, x0, lr=1e-3, lr_auto=False, is_training=True):
        with torch.no_grad():
            s_y0 = torch.matmul(x0, self.w_in.t())
            y0 = self.f(s_y0)
            s_x1 = torch.matmul(y0, self.w_out.t())
            x1 = self.f(s_x1)
            s_y1 = torch.matmul(x1, self.w_in.t())
            # s_x1.register_hook(self.save_grad('s_x1'))
            # s_y1.register_hook(self.save_grad('s_y1'))
            y1 = self.f(s_y1)

            # Gradients
            # y1.sum().backward()
            # with torch.no_grad():
            #     if self.using_derivative:
            #         w_in_grad = torch.matmul(((y1 - y0) * self.grads['s_y1']).t(), x1)
            #         w_out_grad = torch.matmul(((x1 - x0) * self.grads['s_x1']).t(), y0)
            #     else:
            #         w_in_grad = torch.matmul(((y1 - y0) * 1.).t(), x1)
            #         w_out_grad = torch.matmul(((x1 - x0) * 1.).t(), y0)

            if is_training:
                w_in_grad = torch.matmul(((y1 - y0) * self.f_(s_y1)).t(), x1)
                w_out_grad = torch.matmul(((x1 - x0) * self.f_(s_x1)).t(), y0)

                if lr_auto:
                    lr_x, lr_y = self.adaptive_lr_no_bias(x0, y0, s_x1, x1, s_y1, y1)
                    if lr_x == lr_y is None:
                        lr_x, lr_y = (lr, lr)
                else:
                    lr_x, lr_y = (lr, lr)

                # Update weights
                self.w_in = self.w_in - lr_y * w_in_grad
                self.w_out = self.w_out - lr_x * w_out_grad

        return x0, y0, x1, y1

    def _forward_bias(
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
        # self.w_in = self.w_in - torch.matmul(lr_y, w_in_grad)
        # self.t_in = self.t_in - torch.matmul(lr_y, t_in_grad)
        # self.w_out = self.w_out - torch.matmul(lr_x, w_out_grad)
        # self.t_out = self.t_out - torch.matmul(lr_x, t_out_grad)
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
                # if lr_x is not None or lr_y is not None:
                #     lr_x, lr_y = (lr, lr)
            else:
                lr_x, lr_y = (lr, lr)
        else:
            lr_x, lr_y = (lr, lr)
        return lr_x, lr_y

    def adaptive_lr_bias(self, x0, y0, s_x1, x1, s_y1, y1):
        """For LeakyReLU function only"""

        # if not isinstance(self.f, nn.LeakyReLU):
        #     return None, None

        with torch.no_grad():
            rj = self.f_(s_y1)
            ri = self.f_(s_x1)
            rj_next = self.f_(y0)
            ri_next = self.f_(x0)
            bj = (y1 - y0) * rj * (1 + (x1.square()).sum())
            bi = (x1 - x0) * ri * (1 + (y0.square()).sum())
            lr_y = ((s_y1 * rj_next - y0) * bj * rj_next).sum() / (torch.square(rj_next * bj)).sum()
            lr_x = ((s_x1 * ri_next - x0) * bi * ri_next).sum() / (torch.square(ri_next * bi)).sum()
            if self.use_min_max:
                return self.min_max_lr(lr_x), self.min_max_lr(lr_y)
            else:
                return lr_x, lr_y

    def adaptive_lr_bias_every(self, x0, y0, s_x1, x1, s_y1, y1):
        """For LeakyReLU function only"""

        # if not isinstance(self.f, nn.LeakyReLU):
        #     return None, None

        with torch.no_grad():
            rj = self.f_(s_y1)
            ri = self.f_(s_x1)
            rj_next = self.f_(y0)
            ri_next = self.f_(x0)
            bj = (y1 - y0) * rj * (1 + (x1.square()).sum())
            bi = (x1 - x0) * ri * (1 + (y0.square()).sum())
            lr_y = ((s_y1 * rj_next - y0) * bj * rj_next) / (torch.square(rj_next * bj))
            lr_x = ((s_x1 * ri_next - x0) * bi * ri_next) / (torch.square(ri_next * bi))
            if self.use_min_max:
                return self.min_max_lr_every(lr_x), self.min_max_lr_every(lr_y)
            else:
                return lr_x, lr_y

    def forward(
        self,
        x0: torch.Tensor,
        lr: float = 1e-3,
        lr_auto: bool = False,
        lr_const: float = None,
        is_training: bool = True,
        is_bias: bool = True,
    ):
        if is_bias:
            return self._forward_bias(
                x0=x0,
                lr=lr,
                lr_auto=lr_auto,
                lr_const=lr_const,
                is_training=is_training,
            )
        else:
            return self._forward_no_bias(x0, lr, lr_auto, is_training)

    def get_trained_layer(self):
        layer = nn.Linear(in_features=self.in_features, out_features=self.h_features)
        layer.weight.data = self.w_in.data.clone()
        layer.bias.data = self.t_in.data.clone()
        return layer

    def get_bias_out(self):
        return self.t_out.clone()
