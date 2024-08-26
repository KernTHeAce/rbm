import torch


class F_calculator:
    def __init__(self, minuend, subtrahend, factor, activation):
        self.diff = torch.rot90(activation(minuend - subtrahend), 3, [0, 1])
        self.factor = factor

    def __call__(self, cur_obj_i, cur_neuron_i):
        return (torch.flip(self.diff[cur_neuron_i], [0]) * ((self.factor[cur_obj_i] * self.factor).sum(1) + 1)).sum()


class Z_calculator:
    def __init__(self, minuend, subtrahend, factor_1, factor_2, activation):
        self.diff = minuend - subtrahend
        self.factor_1 = torch.rot90(factor_1, 3, [0, 1])
        self.factor_2 = activation(factor_2)

    def __call__(self, cur_obj_i, cur_neuron_i):
        return ((self.diff * self.factor_2[cur_obj_i]).sum(1) * torch.flip(self.factor_1[cur_neuron_i], [0])).sum()


class C_calculator:
    def __init__(self, y_0, y_1, x_0, x_1, s_x, s_y, activation_f):
        self.activation = activation_f
        self.c_j_ = (activation_f(s_y) - y_0)
        self.c_i_ = (activation_f(s_x) - x_0)
        self.f_j_calc = F_calculator(y_1, y_0, x_1, activation_f)
        self.f_i_calc = F_calculator(x_1, x_0, y_0, activation_f)
        self.z_j_calc = Z_calculator(x_1, x_0, y_0, x_1, activation_f)
        self.z_i_calc = Z_calculator(y_1, y_0, x_1, y_0, activation_f)

    def batch_is_over(self, cur_obj_i):
        try:
            self.c_i_[cur_obj_i]
        except IndexError:
            return True
        return False

    def bj(self, cur_obj_i, cur_neuron_i):
        f = self.f_j_calc(cur_obj_i, cur_neuron_i)
        z = self.z_j_calc(cur_obj_i, cur_neuron_i)
        return f + z

    def bi(self, cur_obj_i, cur_neuron_i):
        f = self.f_i_calc(cur_obj_i, cur_neuron_i)
        z = self.z_i_calc(cur_obj_i, cur_neuron_i)
        return f + z

    def cj(self, cur_obj_i, cur_neuron_i):
        return self.c_j_[cur_obj_i][cur_neuron_i] * self.activation(self.bj(cur_obj_i, cur_neuron_i))

    def ci(self, cur_obj_i, cur_neuron_i):
        return self.c_i_[cur_obj_i][cur_neuron_i] * self.activation(self.bi(cur_obj_i, cur_neuron_i))


class AdaptiveLRCalculator:
    def __init__(self, batch_size, input_neurons_num, output_neurons_num):
        self.batch_size = batch_size
        self.input_neurons_num = input_neurons_num
        self.output_neurons_num = output_neurons_num

    def __call__(self, y_0, y_1, x_0, x_1, s_x, s_y, activation_f):
        numerator = 0
        denominator = 0
        c_calculator = C_calculator(y_0, y_1, x_0, x_1, s_x, s_y, activation_f)
        for cur_obj_i in range(self.batch_size):
            if c_calculator.batch_is_over(cur_obj_i):
                return numerator / denominator
            for cur_neuron_i in range(self.output_neurons_num):
                numerator += c_calculator.cj(cur_obj_i, cur_neuron_i)
                denominator += activation_f(c_calculator.bj(cur_obj_i, cur_neuron_i)) ** 2
            for cur_neuron_i in range(self.input_neurons_num):
                numerator += c_calculator.ci(cur_obj_i, cur_neuron_i)
                denominator += activation_f(c_calculator.bi(cur_obj_i, cur_neuron_i)) ** 2
        return numerator / denominator


