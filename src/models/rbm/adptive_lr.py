class AdaptiveLR:
    def __init__(self, batch_size, batch_num):
        self.batch_size = batch_size
        self.batch_num = batch_num
        self.input_neurons_num = input_neurons_num
        self.output_neurons_num = output_neurons_num

    def _calc_c_j(self, y1_all, y0_all, x1_all, activation, cur_neuron_i, cur_batch_i):
        tmp = y1_all[cur_batch_i][cur_neuron_i] - y0_all[cur_batch_i][cur_neuron_i]
        f = self._calc_f_jk(self.input_neurons_num, activation, cur_batch_i, cur_neuron_i, y1_all, y0_all, x1_all)
        z = self._calc_z_jk(self.input_neurons_num, activation, cur_batch_i, cur_neuron_i, y0_all, x1_all, x0_all)
        return tmp * activation(f + z)

    def _calc_c_i(self, y1_all, y0_all, x1_all, activation, cur_neuron_i, cur_batch_i):
        tmp = y1_all[cur_batch_i][cur_neuron_i] - x0_all[cur_batch_i][cur_neuron_i]
        f = self._calc_f_jk(self.output_neurons_num, activation, cur_batch_i, cur_neuron_i, x1_all, x0_all, y0_all)
        z = self._calc_z_jk(self.output_neurons_num, activation, cur_batch_i, cur_neuron_i, x1_all, y1_all, y0_all)
        return tmp * activation(f + z)

    def _calc_f_jk(self, neurons_num, activation, cur_batch_i, cur_neuron_i, y1_all, y0_all, x1_all):
        result = 0
        for item_index in range(self.batch_size):
            tmp = activation(y1_all[item_index][cur_neuron_i] - y0_all[item_index][cur_neuron_i])
            tmp2 = 1
            for neuron_index in neurons_num:
                tmp2 += x1_all[cur_batch_i][neuron_index] * x1_all[item_index][neuron_index]
            result += tmp * tmp2
        return result

    def _calc_z_jk(self, neurons_num, activation, cur_batch_i, cur_neuron_i, y0_all, x1_all, x0_all):
        result = 0
        for item_index in range(self.batch_size):
            tmp = 0
            for neuron_index in neurons_num:
                tmp += activation(x1_all[cur_batch_i][neuron_index]) * (x1_all[item_index][cur_neuron_i] - x0_all[item_index][cur_neuron_i])
            result += tmp * y0_all[item_index][cur_neuron_i]
        return result


    def __call__(self, activation, s_y0, s_y1, s_x1):
        numerator = 0
        for item_index in range(self.batch_size):
            for neuron_index in range(self.output_neurons_num):
                ...

