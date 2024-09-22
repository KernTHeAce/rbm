import torch
from torch import Tensor

from .adaptive_lr import AdaptiveLRCalculator
from .rbm import LayerRBMInitializer


class LayerRbmAdaptiveLrInitializer(LayerRBMInitializer):
    def __init__(
        self,
        layer,
        activation,
        lr,
        grad_min_max,
        t_out: Tensor = None,
        device=torch.device("cuda:0"),
        is_lr_adaptive=True,
        batch_size=None,
        use_grad_clipping=False,
    ):
        super().__init__(layer, activation, lr, t_out, device)
        self.is_lr_adaptive = is_lr_adaptive
        self.lr_calculator = (
            AdaptiveLRCalculator(batch_size, self.in_features, self.out_features) if is_lr_adaptive else None
        )
        self.lr_min_max = (1e-7, 3e-2)
        self.grad_min_max = grad_min_max
        self.use_grad_clipping = use_grad_clipping

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
                    lr = self.lr_calculator(y0, y1, x0, x1, s_x1, s_y1, self.f)
                    lr = min(self.lr_min_max[1], max(lr, self.lr_min_max[0]))
                else:
                    lr = self.lr
                self.update_weights_biases(x0=x0, y0=y0, x1=x1, y1=y1, s_x1=s_x1, s_y1=s_y1, lr=lr)
                if (
                    self.w_in.isnan().sum()
                    or self.w_out.isnan().sum()
                    or self.t_in.isnan().sum()
                    or self.t_out.isnan().sum()
                ):
                    return None
            return x0, y0, x1, y1

    def clip_grad(self, grad):
        if not self.use_grad_clipping:
            return grad
        return torch.clip(grad, min=self.grad_min_max[0], max=self.grad_min_max[1])

    def update_weights_biases(
        self, x0: Tensor, y0: Tensor, x1: Tensor, y1: Tensor, s_x1: Tensor, s_y1: Tensor, lr: float
    ):
        w_in_grad = self.clip_grad(
            torch.matmul((y1 - y0).t(), self.f_(x1)) + torch.matmul((x1 - x0).t(), self.f_(y0)).t()
        )
        t_in_grad = self.clip_grad(self.f_(y1 - y0).sum(dim=0))
        t_out_grad = self.clip_grad(self.f_(x1 - x0).sum(dim=0))
        self.w_in -= lr * w_in_grad
        self.t_in -= lr * t_in_grad
        self.w_out = self.w_in.t()
        self.t_out -= lr * self.clip_grad(t_out_grad)
