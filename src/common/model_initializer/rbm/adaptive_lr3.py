import torch

class AdaptiveLRCalculator:
    def __init__(self, batch_size, input_neurons_num, output_neurons_num):
        self.batch_size = batch_size
        self.input_neurons_num = input_neurons_num
        self.output_neurons_num = output_neurons_num

    @staticmethod
    def _calc_c(S, subtrahend, b, activation_f):
        """
        for c_kj
            S = S_1_kj
            subtrahend = y_0_j
            b = b_kj
        for c_kji
            S = S_1_ki
            subtrahend = x_0_i
            b = b_ki
        """
        return (activation_f(S) - subtrahend) * activation_f(b)

    def calc_b_j(self, y_0, y_1, x_0, x_1, cur_obj_i, cur_neuron_i, activation_f):
        f = self.calc_f(y_1, y_0, x_1, self.input_neurons_num, cur_obj_i, cur_neuron_i, activation_f)
        z = self.calc_z_kj(y_0, y_1, x_0, x_1, cur_obj_i, cur_neuron_i, activation_f)
        return f + z

    def calc_b_i(self, y_0, y_1, x_0, x_1, cur_obj_i, cur_neuron_i, activation_f):
        f = self.calc_f(x_1, x_0, y_0, self.output_neurons_num, cur_obj_i, cur_neuron_i, activation_f)
        z = self.calc_z_ki(y_0, y_1, x_0, x_1, cur_obj_i, cur_neuron_i, activation_f)
        return f + z

    def calc_c_j(self, y_0, y_1, x_0, x_1, s_1, cur_obj_i, cur_neuron_i, activation_f):
        b = self.calc_b_j(y_0, y_1, x_0, x_1, cur_obj_i, cur_neuron_i, activation_f)
        return self._calc_c(s_1[cur_obj_i][cur_neuron_i], y_0[cur_obj_i][cur_neuron_i], b, activation_f)

    def calc_c_i(self, y_0, y_1, x_0, x_1, s_1, cur_obj_i, cur_neuron_i, activation_f):
        b = self.calc_b_i(y_0, y_1, x_0, x_1, cur_obj_i, cur_neuron_i, activation_f)
        return self._calc_c(s_1[cur_obj_i][cur_neuron_i], x_0[cur_obj_i][cur_neuron_i], b, activation_f)

    def calc_f(self, minuend, subtrahend, factor, neurons_num, cur_obj_i, cur_neuron_i, activation_f):
        """
        for f_kj
            minuend = y_1
            subtrahend = y_0
            factor = x_1
            neurons_num = input_neurons_num
        for f_ki
            minuend = x_1
            subtrahend = x_0
            factor = y_0
            neurons_num = output_neurons_num
        """
        result = 0
        for obj_i in range(self.batch_size):
            tmp = activation_f(minuend[obj_i][cur_neuron_i] - subtrahend[obj_i][cur_neuron_i])
            tmp2 = torch.Tensor([0]).to(factor.device)
            for neuron_index in range(neurons_num):
                tmp2 += factor[cur_obj_i][neuron_index] * factor[obj_i][neuron_index]
            result += tmp * tmp2
        return result

    def calc_z_kj(self, y_0, y_1, x_0, x_1, cur_obj_i, cur_neuron_i, activation_f):
        result = torch.Tensor([0]).to(y_0.device)
        for obj_i in range(self.batch_size):
            tmp = torch.Tensor([0]).to(y_0.device)
            for neuron_i in range(self.input_neurons_num):
                tmp += activation_f(x_1[cur_obj_i][neuron_i]) * (x_1[obj_i][neuron_i] - x_0[obj_i][neuron_i])
            result += tmp * y_0[obj_i][cur_neuron_i]
        return result

    def calc_z_ki(self, y_0, y_1, x_0, x_1, cur_obj_i, cur_neuron_i, activation_f):
        result = torch.Tensor([0]).to(y_0.device)
        for obj_i in range(self.batch_size):
            tmp = torch.Tensor([0]).to(y_0.device)
            for neuron_i in range(self.output_neurons_num):
                tmp += activation_f(y_0[cur_obj_i][neuron_i]) * (y_1[obj_i][neuron_i] - y_0[obj_i][neuron_i])
            result += tmp * x_1[obj_i][cur_neuron_i]
        return result

    def __call__(self, y_0, y_1, x_0, x_1, s_x, s_y, activation_f):
        numerator = 0
        denominator = 0
        for cur_obj_i in range(self.batch_size):
            for cur_neuron_i in range(self.output_neurons_num):
                numerator += self.calc_c_j(y_0, y_1, x_0, x_1, s_y, cur_obj_i, cur_neuron_i, activation_f)
                denominator += activation_f(
                    self.calc_b_j(y_0, y_1, x_0, x_1, cur_obj_i, cur_neuron_i, activation_f)
                ) ** 2
            for cur_neuron_i in range(self.input_neurons_num):
                numerator += self.calc_c_i(y_0, y_1, x_0, x_1, s_x, cur_obj_i, cur_neuron_i, activation_f)
                denominator += activation_f(
                    self.calc_b_i(y_0, y_1, x_0, x_1, cur_obj_i, cur_neuron_i, activation_f)
                ) ** 2
        return numerator / denominator


