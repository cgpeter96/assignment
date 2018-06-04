# coding:utf8
'''
name: 王帅琪
id: 2015522114
'''
import numpy as np
from src.mnist_loader import load_data_wrapper
import math


class Network:
    # 神经元节点
    class Unit:
        # 上次结果的缓存
        _cache_x = None
        _cache_y = None

        def __init__(self, rate, w, b=0):
            self.w, self.rate, self.b = w, rate, b

        # 误差调整函数
        def reset(self, label):
            sigma = self._cache_y * (1 - self._cache_y) * (label - self._cache_y)
            # 更新每个权值
            self.w += self._cache_x.T * sigma
            self.b += sigma * self.rate
            return sigma

        # 激励函数
        def output(self, x):
            # 缓存结果以备调整
            self._cache_x = x
            self._cache_y = 1 / (1 + math.exp(-(x * self.w + self.b)))
            return self._cache_y

    # 获得初始权值矩阵
    def _get_weight(self, size):
        return np.mat(
            np.random.uniform(-2.4 / self.input_num, 2.4 / self.output_num, size=size))

    # 获得初始神经元列表
    def _get_n_list(self, num, w=None):
        return [self.Unit(self.rate, w) for _ in range(num)]

    def __init__(self, input_num, output_num, middle_num=90, rate=0.1):
        self.input_num, self.output_num, self.middle_num, self.rate = input_num, output_num, middle_num, rate

        # 输入层神经元列表
        self._input = self._get_n_list(input_num, w=self._get_weight((1, 1)))
        # 隐含层神经元列表
        self._middle = self._get_n_list(middle_num, w=self._get_weight((input_num, 1)))
        # 输出层神经元列表
        self._output = self._get_n_list(output_num, w=self._get_weight((middle_num, 1)))

    def train(self, train_label):
        # 反向传播，上层节点目标值
        middle_label_list = []

        for each_output_unit, each_label in zip(self._output, train_label):
            sigma = each_output_unit.reset(each_label)
            middle_label_list.append((sigma, each_output_unit.w))

        for index, each_middle_unit in enumerate(self._middle):
            label = 0
            for each_sigma, each_w in middle_label_list:
                label += each_w[index] * each_sigma
            each_middle_unit.reset(label)

    def output(self, input_data):
        input_output = np.mat([input_unit.output(input_x) for input_unit, input_x in zip(self._input, input_data)])
        middle_output = np.mat([middle_unit.output(input_output) for middle_unit in self._middle])
        return [output_unit.output(middle_output) for output_unit in self._output]


if __name__ == '__main__':
    INPUT_SIGNAL = 784
    OUTPUT_SIGNAL = 10
    MIDDLE_SIGNAL = 3


    def get_max_index(arr):
        tmp = [item for item in arr]
        return tmp.index(max(tmp))


    n = Network(INPUT_SIGNAL, OUTPUT_SIGNAL, MIDDLE_SIGNAL)
    training_data, validation_data, test_data = load_data_wrapper()

    # while True:
    #     flag = False  # 未出现误差点
    #     counter = 0
    #     for each in training_data:
    #         counter += 1
    #         result = n.output(each[0])
    #         if not get_max_index(result) == get_max_index(each[1]):
    #             n.train(each[1])
    #             flag = True
    #             break
    #         else:
    #             print(counter)
    #     if not flag:
    #         break
