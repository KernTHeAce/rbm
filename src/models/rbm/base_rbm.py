from torch import nn


class BaseRBM(nn.Module):
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

    def forward(self, *args, **kwargs):
        pass

    def get_trained_layer(self):
        pass

    def get_bias_out(self):
        pass


def output_padding_conv_transpose2d(layer: nn.Conv2d, input_shape):
    assert len(input_shape) == 4
    assert layer.kernel_size[0] == layer.kernel_size[1]
    assert layer.stride[0] == layer.stride[1]
    assert layer.padding[0] == layer.padding[1]
    assert layer.dilation[0] == layer.dilation[1]

    i = input_shape[3]
    k = layer.kernel_size[0]
    s = layer.stride[0]
    p = layer.padding[0]
    d = layer.dilation[0]

    o = int((i + 2 * p - k - (k - 1) * (d - 1)) / s + 1)
    op = i - (o - 1) * s + 2 * p - k

    return op


def output_padding_conv_transpose1d(layer: nn.Conv1d, input_shape):
    assert len(input_shape) == 3

    i = input_shape[2]
    k = layer.kernel_size[0]
    s = layer.stride[0]
    p = layer.padding[0]
    d = layer.dilation[0]

    o = int((i + 2 * p - k - (k - 1) * (d - 1)) / s + 1)
    op = i - (o - 1) * s + 2 * p - k

    return op
