from torch import nn


class BaseRBM:
    def __init__(self, *args, **kwargs):
        ...

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
