#!usr/bin/python
# coding=utf8

import numpy as np
from scipy.interpolate import griddata

from mcts import DesignSpace, SampleBag, FunctionBase


class Replayer(FunctionBase):
    def __init__(
        self,
        name,
        design_space: DesignSpace,
        sample_bag: SampleBag,
        verbose,
        lazy,
        data_path,
        method,
    ):
        super().__init__(name, design_space, sample_bag, verbose=verbose, lazy=lazy)
        self.data_path = data_path
        self.method = method
        self.data = SampleBag(self.ds)

    def load_data(self):
        self.data.load_csv2(self.data_path)
        self.X = self.data.getX()
        self.Y = -self.data.getY()

    def load_data2(self):
        self.data.load_csv2(self.data_path)
        # print(self.data.data)
        # print(self.data.data.describe())
        self.X = self.data.getX()
        self.Y = self.data.getY()

    def calculate(self, x):
        y = griddata(self.X, self.Y, x.reshape(1, -1), self.method, rescale=True)
        return y.item()

    def calculate_batch(self, X):
        Y = griddata(self.X, self.Y, X, self.method, rescale=True)
        return Y

    def exe_batch(self, X):
        if self.lazy:
            for x in X:
                self.validate_input(x)
            Y = self.calculate_batch(X)
            self.track(X, Y)
            return -Y if self.minimize_mode else Y
        else:
            Y = []
            for x in X:
                Y.append(self.__call__(x))
            Y = np.array(Y)
            return Y
